import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import argparse
import numpy as np
from tqdm import tqdm

# --- Clean, Corrected Imports ---
from aps import crop_image
from dts import dts
from model import DocKylinQwen
from dataset import FUNSDInstructionDataset

# --- Main Fine-tuning Function ---
def main(args):
    # 1. Instantiate Model from model.py
    model = DocKylinQwen()
    
    # 2. Load the Swin+Qwen pre-trained checkpoint
    print(f"--- Loading Swin+Qwen pre-trained weights from {args.pretrained_checkpoint} ---")
    model.load_state_dict(torch.load(args.pretrained_checkpoint))
    model.train() 
    
    # 3. Ensure all components are on the correct device and dtype
    device = model.language_model.device
    model.visual_encoder.to(device)
    model.mlp.to(device=device, dtype=model.language_model.dtype)
    print(f"--- Model components moved to device: {device} and dtype: {model.language_model.dtype} ---")

    # 4. Unfreeze all model parameters for full fine-tuning
    print("--- Unfreezing all model parameters for full fine-tuning. ---")
    for param in model.parameters():
        param.requires_grad = True
    
    # 5. Setup Optimizer to train ALL model parameters
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 6. Load the Instruction Dataset
    dataset = FUNSDInstructionDataset(split='train')
    def collate_fn(batch): 
        return batch[0]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    print("--- Starting Full Instruction Fine-tuning with Swin+Qwen+DTS ---")
    for epoch in range(args.epochs):
        # Wrap the dataloader with tqdm for a progress bar
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            optimizer.zero_grad()

            # --- Prepare Inputs ---
            pil_image = batch['image']
            instruction = batch['instruction']
            answer = batch['answer']

            # Image Processing with APS
            cv2_image = np.array(pil_image.convert("RGB"))
            cv2_image = cv2_image[:, :, ::-1].copy()
            cropped_cv2_image, _, _ = crop_image(cv2_image, 'xy')
            cropped_pil_image = Image.fromarray(cropped_cv2_image[:, :, ::-1])
            
            # Use the official image processor
            pixel_values = model.image_processor(cropped_pil_image, return_tensors="pt").pixel_values.to(device)

            # --- Visual Feature Extraction and DTS ---
            visual_features = model.visual_encoder(pixel_values).last_hidden_state
            
            _, essential_idx, _, _ = dts(visual_features, visual_features)
            squeezed_idx = essential_idx.flatten()
            
            index = squeezed_idx.unsqueeze(0).unsqueeze(-1).expand(visual_features.shape[0], -1, visual_features.shape[-1])
            essential_features = torch.gather(visual_features, 1, index)
            
            projected_features = model.mlp(essential_features.to(model.language_model.dtype))
            
            # --- Text and Label Preparation ---
            messages = [{"role": "user", "content": instruction}]
            prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            full_text = prompt + answer + model.tokenizer.eos_token
            full_tokens = model.tokenizer(full_text, return_tensors='pt', max_length=1024, truncation=True).to(device)
            
            labels = full_tokens.input_ids.clone()
            prompt_len = len(model.tokenizer(prompt, return_tensors='pt').input_ids[0])
            labels[0, :prompt_len] = -100

            # --- Forward Pass for Loss Calculation ---
            text_embeddings = model.language_model.get_input_embeddings()(full_tokens.input_ids)
            combined_embeddings = torch.cat([projected_features, text_embeddings], dim=1)
            
            visual_attention_mask = torch.ones(projected_features.shape[:2], dtype=torch.long, device=device)
            combined_attention_mask = torch.cat([visual_attention_mask, full_tokens.attention_mask], dim=1)
            
            visual_labels = torch.full(projected_features.shape[:2], -100, dtype=torch.long, device=device)
            combined_labels = torch.cat([visual_labels, labels], dim=1)

            outputs = model.language_model(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_attention_mask,
                labels=combined_labels
            )
            loss = outputs.loss
            
            # --- Backward Pass and Optimization ---
            if loss is not None:
                loss.backward()
                optimizer.step()
                # Log the loss every 10 steps to see progress
                if (i + 1) % 10 == 0:
                    tqdm.write(f"Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        print(f"--- Epoch {epoch+1} finished. Saving checkpoint... ---")
        torch.save(model.state_dict(), f"dockylin_swin_qwen_finetuned_epoch_{epoch+1}.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DocKylin Full Fine-tuning with Swin+Qwen+DTS')
    parser.add_argument('--pretrained_checkpoint', type=str, required=True, help='Path to the Swin+Qwen-compatible PRE-TRAINED checkpoint.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of fine-tuning epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='A smaller learning rate for fine-tuning.')
    args = parser.parse_args()
    main(args)
