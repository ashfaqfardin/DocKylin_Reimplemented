import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor, SwinModel

class DocKylinQwen(nn.Module):
    """
    This class defines the final model architecture.
    It combines a Swin Transformer vision encoder with a Qwen language model.
    """
    def __init__(self, 
                 vision_encoder_name='microsoft/swin-tiny-patch4-window7-224',
                 lang_model_name='Qwen/Qwen1.5-1.8B-Chat'):
        super().__init__()
        print(f"--- Initializing Vision Encoder: {vision_encoder_name} ---")
        print(f"--- Initializing Language Model: {lang_model_name} ---")

        # 1. Vision Encoder (Swin Transformer)
        self.visual_encoder = SwinModel.from_pretrained(vision_encoder_name)
        self.image_processor = AutoImageProcessor.from_pretrained(vision_encoder_name)

        # 2. MLP Projection Layer
        # The Swin 'tiny' model outputs features of dimension 768.
        vision_feature_dim = 768 
        # The Qwen '1.8B' model expects inputs of dimension 2048.
        lang_embedding_dim = 2048 
        self.mlp = nn.Sequential(
            nn.Linear(vision_feature_dim, lang_embedding_dim),
            nn.GELU(),
            nn.Linear(lang_embedding_dim, lang_embedding_dim)
        )

        # 3. Language Model (Qwen)
        self.language_model = AutoModelForCausalLM.from_pretrained(
            lang_model_name,
            torch_dtype="auto",
            device_map="auto" # Automatically uses GPU if available
        )
        self.tokenizer = AutoTokenizer.from_pretrained(lang_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
