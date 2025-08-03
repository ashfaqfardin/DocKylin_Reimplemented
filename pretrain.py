import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import argparse
import numpy as np
from tqdm import tqdm

from aps import crop_image
from model import DocKylinQwen
from dataset import FUNSDDataset

def main(args):
    # 1. Instantiate the new model with Swin + Qwen
    model = DocKylinQwen()
    model.train()

    # 2. Move components to the correct device and dtype
    device = model.language_model.device
    model.visual_encoder.to(device)
    model.mlp.to(device=device, dtype=model.language_model.dtype)
    print(f"--- Model components moved to device: {device} ---")

    # 3. Freeze the Qwen LLM for pre-training
    for param in model.language_model.parameters():
        param.requires_grad = False
    
    # 4. Setup optimizer to train only the vision encoder and MLP
    optimizer = optim.AdamW([
        {'params': model.visual_encoder.parameters()},
        {'params': model.mlp.parameters()}
    ], lr=args.learning_rate)

    # 5. Load dataset and dataloader
    dataset = FUNSDDataset(split='train')
    def collate_fn(batch):
        return batch[0]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    print("--- Starting Pre-training with Swin Encoder for Qwen Base ---")
    for epoch in range(args.epochs):
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            optimizer.zero_grad()
            
            # --- Prepare Inputs ---
            pil_image = batch['image']
            text = batch['text']

            # Image processing with APS
            cv2_image = np.array(pil_image.convert("RGB"))
            cv2_image = cv2_image[:, :, ::-1].copy()
            cropped_cv2_image, _, _ = crop_image(cv2_image, 'xy')
            cropped_pil_image = Image.fromarray(cropped_cv2_image[:, :, ::-1])
            
            # Use the official image processor for the Swin model
            pixel_values = model.image_processor(cropped_pil_image, return_tensors="pt").pixel_values.to(device)

            # --- Forward Pass ---
            visual_features = model.visual_encoder(pixel_values).last_hidden_state
            avg_visual_features = torch.mean(visual_features, dim=1)
            
            projected_features = model.mlp(avg_visual_features.to(model.language_model.dtype))
            
            # Prepare text tokens
            tokens = model.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            text_embeddings = model.language_model.get_input_embeddings()(tokens.input_ids)

            # Combine embeddings
            combined_embeddings = torch.cat([projected_features.unsqueeze(1), text_embeddings], dim=1)
            
            # Create attention mask and labels
            visual_attention_mask = torch.ones((1,1), dtype=torch.long, device=device)
            combined_attention_mask = torch.cat([visual_attention_mask, tokens.attention_mask], dim=1)
            labels = tokens.input_ids.clone()
            visual_labels = torch.full((1, 1), -100, dtype=torch.long, device=device)
            combined_labels = torch.cat([visual_labels, labels], dim=1)

            # Calculate Loss
            outputs = model.language_model(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_attention_mask,
                labels=combined_labels
            )
            loss = outputs.loss
            
            # --- Backward Pass ---
            if loss is not None:
                loss.backward()
                optimizer.step()
                # FIX: Log the loss every 10 steps for more frequent updates
                if (i + 1) % 10 == 0:
                    tqdm.write(f"Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        # Save a new Swin-based checkpoint
        torch.save(model.state_dict(), f"dockylin_swin_qwen_pretrain_epoch_{epoch+1}.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DocKylin Pre-training with Swin+Qwen')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate.')
    args = parser.parse_args()
    main(args)
