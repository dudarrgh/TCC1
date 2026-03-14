from typing import Dict, List, Any
from diffusers import DiffusionPipeline, AutoencoderKL
import torch
import os
from io import BytesIO
from PIL import Image

class EndpointHandler():
    def __init__(self, path=""):
        # Carrega o modelo base (SDXL ou SD1.5 - ajuste conforme o que você usou no Kohya)
        self.pipeline = DiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float16
        ).to("cuda")
        
        # Carrega o seu arquivo LoRA que você subiu
        self.pipeline.load_lora_weights(path, weight_name="estilo_pedagogico_v2.safetensors")

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        inputs = data.pop("inputs", data)
        
        # Gera a imagem usando o seu estilo
        image = self.pipeline(inputs, num_inference_steps=30).images[0]
        
        return image