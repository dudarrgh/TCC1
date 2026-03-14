---
license: creativeml-openrail-m
base_model: stabilityai/stable-diffusion-xl-base-1.0
instance_prompt: estilo_pedagogico
tags:
- text-to-image
- diffusers
- lora
- template:sd-lora
widget:
- text: "estilo_pedagogico, a cute owl reading a book, flat design"
  output:
    url: "output.png"
---

# Modelo de Cartões Pedagógicos (LoRA)

Este modelo foi desenvolvido como parte de um TCC focado em IA Assistiva Infantil. 
Ele foi treinado para gerar ilustrações com traços grossos, cores sólidas e estética de "Flat Design".

## Detalhes do Treinamento
- **Hardware:** NVIDIA RTX GPU
- **Steps:** 1600
- **Trigger Word:** estilo_pedagogico
- **Dataset:** 20 imagens curadas de alta qualidade.