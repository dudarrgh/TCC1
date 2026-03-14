import torch
from diffusers import StableDiffusionPipeline

# 1. Carrega o modelo base
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda") # Usa sua RTX 3050

# 2. Carrega o seu estilo (O arquivo tem que estar na mesma pasta!)
pipe.load_lora_weights(".", weight_name="estilo_pedagogico_v2.safetensors")

# 3. Prompt usando sua palavra-chave
prompt = "A cute lion, estilo_pedagogico, flat design, thick outlines, white background"

print("🎨 Gerando imagem com o SEU estilo treinado...")
image = pipe(prompt, num_inference_steps=30).images[0]

# 4. Salva o resultado
image.save("resultado_tcc.png")
image.show()
print("✅ Sucesso! Verifique o arquivo resultado_tcc.png na pasta.")