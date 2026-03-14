import streamlit as st
import torch
import os
import io
import re
import random
from PIL import Image
from deep_translator import GoogleTranslator
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

# ============================================================
# CONFIGURAÇÃO (EVITA CUDA OUT OF MEMORY)
# ============================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================
# CONFIG FIXA (NÃO APARECE NA TELA)
# ============================================================
STEPS_FIXO = 45
GUIDANCE_FIXO = 11.0
IMG_SIZE = 512

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def normalize_user_prompt_pt(text: str) -> str:
    """
    Normaliza o input pra evitar que o SD tente gerar texto.
    Ex: "dizendo sim" -> remove "dizendo"
    """
    t = text.strip()

    replacements = [
        (r"\bdizendo\b", ""),
        (r"\bescrito\b", ""),
        (r"\btexto\b", ""),
        (r"\bfrase\b", ""),
        (r"\bcom a palavra\b", ""),
        (r"\bescrevendo\b", ""),
    ]

    for pattern, repl in replacements:
        t = re.sub(pattern, repl, t, flags=re.IGNORECASE)

    t = re.sub(r"\s+", " ", t).strip()
    return t


def detect_mode(prompt_pt: str) -> str:
    """
    Detecta se o pedido é mais provável ser:
    - 'animal' (gatinho, cachorro, etc.)
    - 'human' (menino, menina, criança, etc.)
    Default: human (porque você quer pessoas no dataset)
    """
    t = prompt_pt.lower()

    animal_keywords = [
        "gato", "gatinho", "gata", "cachorro", "cão", "cao", "coelho", "pássaro", "passaro",
        "peixe", "tartaruga", "urso", "panda", "leão", "leao", "macaco", "girafa", "elefante",
        "dinossauro", "dinossaurinho", "patinho", "pintinho"
    ]
    human_keywords = [
        "menino", "menina", "criança", "crianca", "bebê", "bebe", "garoto", "garota",
        "pessoa", "menininho", "menininha", "criançinha", "criancinha"
    ]

    has_animal = any(k in t for k in animal_keywords)
    has_human = any(k in t for k in human_keywords)

    if has_animal and not has_human:
        return "animal"
    return "human"


def build_prompts(prompt_en: str, mode: str):
    """
    Cria prompt positivo e negativo.
    mode:
      - 'human': força criança/pessoa e bloqueia animal
      - 'animal': força animal e bloqueia humano
    """

    # --------------------------
    # Conteúdo (o que o usuário quer)
    # --------------------------
    content_prompt = (
        f"{prompt_en}, "
        "the action must be clearly visible, "
        "single scene, simple"
    )

    # --------------------------
    # Estilo (igual ao seu dataset)
    # --------------------------
    style_prompt = (
        "educational AAC communication card, preschool kids illustration, "
        "centered single subject, close-up, zoomed in, tight framing, "
        "character very large and fills the frame, no empty space, "
        "cute kawaii chibi style, big shiny eyes, rosy cheeks, friendly smile, "
        "thick bold rounded outlines, clean smooth lineart, "
        "flat 2D illustration, vibrant solid colors, very soft shading, "
        "soft pastel background with subtle bokeh dots and sparkles, bright lighting, "
        "high visual clarity, minimal background details, no clutter, "
        "estilo_pedagogico"
    )

    # ✅ Conteúdo primeiro (para obedecer melhor)
    prompt_final = f"{content_prompt}, {style_prompt}"

    # --------------------------
    # Negative base (sempre)
    # --------------------------
    negative_base = (
        "letters, numbers, text, typography, words, watermark, logo, label, signage, caption, "
        "sonic, sega, mario, nintendo, disney, pixar, nickelodeon, cartoon network, "
        "video game character, famous character, franchise character, "
        "photorealistic, realistic, 3d render, anime, semi-realistic, "
        "busy background, cluttered scene, complex scenery, many objects, "
        "dark lighting, harsh shadows, strong gradients, "
        "low quality, blurry, noise, grainy, jpeg artifacts, "
        "creepy, horror, scary, "
        "extra limbs, extra fingers, deformed, mutated, bad anatomy"
    )

    # --------------------------
    # Negative e reforços por modo
    # --------------------------
    if mode == "animal":
        # Força animal e evita humano aparecer
        prompt_boost = "cute kitten, cute cat, animal character, "
        negative_mode = "human, kid, child, boy, girl, baby, person, people, "
        prompt_final = prompt_boost + prompt_final
    else:
        # Força criança/pessoa e evita animal aparecer
        prompt_boost = "cute little boy or cute little girl, child character, "
        negative_mode = "animal, cat, dog, kitten, puppy, "
        prompt_final = prompt_boost + prompt_final

    negative_prompt = negative_mode + negative_base
    return prompt_final, negative_prompt


# ============================================================
# MODELO LOCAL
# ============================================================

@st.cache_resource
def carregar_modelo_local():
    model_id = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None
    )

    # Otimizações para 4GB
    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing()

    # Scheduler melhor
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # LoRA criado
    pipe.load_lora_weights(".", weight_name="estilo_pedagogico_v2.safetensors")

    return pipe


# ============================================================
# UI (LIMPA)
# ============================================================

st.set_page_config(page_title="IA Assistiva - TCC Eduarda", layout="centered")
st.title("🎨 Gerador Pedagógico Local")
st.subheader("Processamento Otimizado para RTX 3050 (4GB)")

# Inicialização do modelo
try:
    if "pipe" not in st.session_state:
        with st.status("Preparando motor da IA... (otimizado pra memória)", expanded=True):
            st.session_state.pipe = carregar_modelo_local()
        st.success("✅ Sistema pronto, Eduarda!")
except Exception as e:
    st.error(f"Erro crítico de memória: {e}. Dica: feche abas extras e reinicie o app.")

# Entrada
prompt_pt = st.text_input(
    "O que vamos desenhar hoje?",
    placeholder="Ex: gatinho bebendo água / menininho comendo / criança escovando os dentes"
)

# Botão gerar
if st.button("✨ Gerar Cartão Pedagógico"):
    if not prompt_pt:
        st.warning("⚠️ Digite uma descrição primeiro.")
    else:
        container = st.empty()
        try:
            # 1) Normaliza prompt
            prompt_pt_norm = normalize_user_prompt_pt(prompt_pt)

            # 2) Detecta modo automaticamente (human/animal)
            mode = detect_mode(prompt_pt_norm)

            # 3) Tradução
            translator = GoogleTranslator(source="pt", target="en")
            prompt_en = translator.translate(prompt_pt_norm)

            # 4) Prompts
            prompt_final, negative_prompt = build_prompts(prompt_en, mode)

            # 5) Seed ALEATÓRIA (SEM seed fixa)
            seed = random.randint(0, 9999999)
            generator = torch.Generator(device="cuda").manual_seed(seed)

            # 6) Geração
            torch.cuda.empty_cache()
            container.info("🎨 Renderizando...")

            image = st.session_state.pipe(
                prompt=prompt_final,
                negative_prompt=negative_prompt,
                width=IMG_SIZE,
                height=IMG_SIZE,
                num_inference_steps=STEPS_FIXO,
                guidance_scale=GUIDANCE_FIXO,
                generator=generator
            ).images[0]

            # 7) Exibição + download
            container.empty()
            st.image(image, caption=f"Resultado para: {prompt_pt}", use_container_width=True)

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            st.download_button(
                "📥 Baixar Cartão",
                buf.getvalue(),
                "cartao_eduarda.png",
                "image/png"
            )

        except Exception as e:
            st.error(f"Erro: {e}")
            torch.cuda.empty_cache()

st.markdown("---")
st.caption("Protótipo de TCC - Eduarda 2026 | Stable Diffusion v1.5 + LoRA (estilo_pedagogico_v2)")
