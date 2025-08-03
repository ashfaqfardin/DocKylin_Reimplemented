import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import argparse
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import re
import json

# --- Clean, Corrected Imports ---
from aps import crop_image
from dts import dts
from model import DocKylinQwen
from dataset import FUNSDInstructionDataset

# --- Accuracy Metric Calculation ---
def normalize_text(s):
    """Lowercase, remove articles, punctuation, and extra whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def accuracy_score(prediction, ground_truth):
    """Calculates exact match accuracy after normalization."""
    return 1 if normalize_text(prediction) == normalize_text(ground_truth) else 0

# --- Main Evaluation Function ---
def evaluate(args):
    # 1. Instantiate Model and load checkpoint
    model = DocKylinQwen()
    print(f"--- Loading fine-tuned weights from {args.finetuned_checkpoint} ---")
    model.load_state_dict(torch.load(args.finetuned_checkpoint))
    model.eval()

    # 2. Move all model components to the correct device and dtype
    device = model.language_model.device
    model.visual_encoder.to(device)
    model.mlp.to(device=device, dtype=model.language_model.dtype)
    print(f"--- Model components moved to device: {device} and dtype: {model.language_model.dtype} ---")

    # 3. Load the test dataset
    dataset = FUNSDInstructionDataset(split='test')
    
    correct_predictions = 0
    evaluation_results = []
    
    print("--- Starting Evaluation for Accuracy (with APS & DTS) ---")
    
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        
        with torch.no_grad():
            # --- Image Processing (with APS) ---
            pil_image = item['image']
            cv2_image = np.array(pil_image.convert("RGB"))
            cv2_image = cv2_image[:, :, ::-1].copy()
            cropped_cv2_image, _, _ = crop_image(cv2_image, 'xy')
            cropped_pil_image = Image.fromarray(cropped_cv2_image[:, :, ::-1])
            
            # Use the official image processor for the Swin model
            pixel_values = model.image_processor(cropped_pil_image, return_tensors="pt").pixel_values.to(device)

            # --- Visual Feature Extraction and DTS ---
            # Correctly get patch embeddings from the Swin encoder
            visual_features = model.visual_encoder(pixel_values).last_hidden_state
            
            # Apply DTS to select only the most essential tokens
            _, essential_idx, _, _ = dts(visual_features, visual_features)
            squeezed_idx = essential_idx.flatten()
            
            index = squeezed_idx.unsqueeze(0).unsqueeze(-1).expand(visual_features.shape[0], -1, visual_features.shape[-1])
            essential_features = torch.gather(visual_features, 1, index)
            
            # Project the SLIMMED features through the MLP
            projected_features = model.mlp(essential_features.to(model.language_model.dtype))
            
            # --- Prepare Prompt and Generate ---
            instruction = item['instruction']
            messages = [{"role": "user", "content": instruction}]
            prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_tokens = model.tokenizer(prompt, return_tensors='pt').to(device)
            prompt_embeddings = model.language_model.get_input_embeddings()(prompt_tokens.input_ids)

            # Combine the SLIMMED visual embeddings with text embeddings
            combined_embeddings = torch.cat([projected_features, prompt_embeddings], dim=1)
            
            # Create the attention mask for the new, shorter sequence
            visual_attention_mask = torch.ones(projected_features.shape[:2], dtype=torch.long, device=device)
            combined_attention_mask = torch.cat([visual_attention_mask, prompt_tokens.attention_mask], dim=1)

            # --- Generate Answer ---
            output_ids = model.language_model.generate(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_attention_mask,
                max_new_tokens=150,
                min_new_tokens=1, # Prevents empty predictions
                pad_token_id=model.tokenizer.eos_token_id
            )
            
            # Decode the output, slicing off the full prompt length
            start_index = combined_embeddings.shape[1]
            generated_answer = model.tokenizer.decode(output_ids[0, start_index:], skip_special_tokens=True)
            
        # --- Score and Store ---
        ground_truth_answer = item['answer']
        is_correct = accuracy_score(generated_answer, ground_truth_answer)
        correct_predictions += is_correct
        evaluation_results.append({
            "instruction": instruction,
            "ground_truth": ground_truth_answer,
            "prediction": generated_answer,
            "is_correct": bool(is_correct)
        })

    # --- Save and Print Final Results ---
    with open(args.output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"\n--- Detailed results saved to {args.output_file} ---")
    
    final_accuracy = (correct_predictions / len(dataset)) * 100
    print(f"Final Accuracy (with APS & DTS): {final_accuracy:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DocKylin Accuracy Evaluation with APS and DTS')
    parser.add_argument('--finetuned_checkpoint', type=str, required=True, help='Path to your Swin+Qwen fine-tuned model checkpoint.')
    parser.add_argument('--output_file', type=str, default='evaluation_results_with_slimming.json', help='File to save the evaluation results.')
    args = parser.parse_args()
    evaluate(args)
