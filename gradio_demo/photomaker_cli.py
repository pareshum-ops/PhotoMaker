#!/usr/bin/env python3
"""
PhotoMaker V2 CLI - Generate images without Gradio
Just edit the configuration below and run: python photomaker_cli.py
"""

import torch
import torchvision.transforms.functional as TF
import numpy as np
import random
import os
import sys
from pathlib import Path

from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler, T2IAdapter
from huggingface_hub import hf_hub_download

from photomaker import PhotoMakerStableDiffusionXLAdapterPipeline
from photomaker import FaceAnalysis2, analyze_faces

from style_template import styles
from aspect_ratio_template import aspect_ratios

# ============================================================
# CONFIGURATION - Edit these values directly
# ============================================================

# Input image(s) - provide path(s) to face image(s)
INPUT_IMAGES = [
    "/teamspace/studios/this_studio/PhotoMaker/Data/Input/Rafa&John_1.webp",
    # "./input/face2.jpg",  # Add more images for better ID fidelity
]

# Prompt - must include 'img' trigger word
PROMPT = "a photo of two people img, both faces clearly visible, both wearing eyeglasses, natural fit, realistic reflections on lenses, sharp facial details, preserved identity, same pose and background, high quality"

# Output settings
OUTPUT_DIR = "/teamspace/studios/this_studio/PhotoMaker/Data/Output"
NUM_OUTPUTS = 2

# Style (check style_template.py for options)
STYLE_NAME = "Photographic (Default)"

# Negative prompt
NEGATIVE_PROMPT = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

# Output dimensions
OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 1024

# Generation parameters
NUM_STEPS = 50
GUIDANCE_SCALE = 5.0
STYLE_STRENGTH_RATIO = 20
SEED = None  # Set to None for random seed, or specify a number

# Sketch/Doodle settings (optional)
USE_SKETCH = False
SKETCH_IMAGE_PATH = None  # e.g., "./sketch.png"
ADAPTER_CONDITIONING_SCALE = 0.7
ADAPTER_CONDITIONING_FACTOR = 0.8

# ============================================================
# END OF CONFIGURATION
# ============================================================

MAX_SEED = np.iinfo(np.int32).max


def get_device():
    try:
        if torch.cuda.is_available():
            return "cuda"
        elif sys.platform == "darwin" and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except:
        return "cpu"


def apply_style(style_name, positive, negative=""):
    default_style = "Photographic (Default)"
    p, n = styles.get(style_name, styles[default_style])
    return p.replace("{prompt}", positive), n + ' ' + negative


def load_pipeline(device):
    print("Loading pipeline...")
    
    base_model_path = 'SG161222/RealVisXL_V4.0'
    
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        torch_dtype = torch.float16
    
    print("Loading T2I adapter...")
    adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-sketch-sdxl-1.0", 
        torch_dtype=torch_dtype, 
        variant="fp16"
    ).to(device)
    
    print("Loading main pipeline...")
    pipe = PhotoMakerStableDiffusionXLAdapterPipeline.from_pretrained(
        base_model_path, 
        adapter=adapter, 
        torch_dtype=torch_dtype,
        use_safetensors=True, 
        variant="fp16",
    ).to(device)
    
    print("Loading PhotoMaker adapter...")
    photomaker_ckpt = hf_hub_download(
        repo_id="TencentARC/PhotoMaker-V2", 
        filename="photomaker-v2.bin", 
        repo_type="model"
    )
    
    pipe.load_photomaker_adapter(
        os.path.dirname(photomaker_ckpt),
        subfolder="",
        weight_name=os.path.basename(photomaker_ckpt),
        trigger_word="img",
        pm_version="v2",
    )
    pipe.id_encoder.to(device)
    
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.fuse_lora()
    pipe.to(device)
    
    print("Pipeline loaded successfully!")
    return pipe


def load_face_detector(device):
    print("Loading face detector...")
    providers = ['CUDAExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
    face_detector = FaceAnalysis2(
        providers=providers, 
        allowed_modules=['detection', 'recognition']
    )
    face_detector.prepare(ctx_id=0, det_size=(640, 640))
    return face_detector


def generate_image(pipe, face_detector, device):
    # Handle sketch input
    sketch_image = None
    adapter_scale = 0.
    adapter_factor = 0.
    
    if USE_SKETCH and SKETCH_IMAGE_PATH:
        from PIL import Image
        sketch_image = Image.open(SKETCH_IMAGE_PATH).convert("RGBA")
        r, g, b, a = sketch_image.split()
        sketch_image = a.convert("RGB")
        sketch_image = TF.to_tensor(sketch_image) > 0.5
        sketch_image = TF.to_pil_image(sketch_image.to(torch.float32))
        adapter_scale = ADAPTER_CONDITIONING_SCALE
        adapter_factor = ADAPTER_CONDITIONING_FACTOR

    # Check trigger word
    image_token_id = pipe.tokenizer.convert_tokens_to_ids(pipe.trigger_word)
    input_ids = pipe.tokenizer.encode(PROMPT)
    if image_token_id not in input_ids:
        raise ValueError(f"Cannot find trigger word '{pipe.trigger_word}' in prompt! Include 'img' in your prompt.")

    if input_ids.count(image_token_id) > 1:
        raise ValueError(f"Cannot use multiple trigger words '{pipe.trigger_word}' in prompt!")

    output_w = OUTPUT_WIDTH or 1024
    output_h = OUTPUT_HEIGHT or 1024
    
    print(f"[Info] Output dimensions: {output_w} x {output_h}")

    # Apply style
    prompt, negative_prompt = apply_style(STYLE_NAME, PROMPT, NEGATIVE_PROMPT)

    # Load input images
    if not INPUT_IMAGES:
        raise ValueError("No input images! Edit INPUT_IMAGES in the script.")

    input_id_images = []
    for img_path in INPUT_IMAGES:
        if not os.path.exists(img_path):
            raise ValueError(f"Image not found: {img_path}")
        input_id_images.append(load_image(img_path))
    
    # Extract face embeddings
    id_embed_list = []
    for img in input_id_images:
        img_array = np.array(img)
        img_array = img_array[:, :, ::-1]
        faces = analyze_faces(face_detector, img_array)
        if len(faces) > 0:
            id_embed_list.append(torch.from_numpy((faces[0]['embedding'])))

    if len(id_embed_list) == 0:
        raise ValueError("No face detected! Use images with clear faces.")
    
    id_embeds = torch.stack(id_embed_list)

    # Handle seed
    seed = SEED if SEED is not None else random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)

    print("Starting inference...")
    print(f"[Info] Seed: {seed}")
    print(f"[Info] Prompt: {prompt}")
    
    start_merge_step = int(float(STYLE_STRENGTH_RATIO) / 100 * NUM_STEPS)
    if start_merge_step > 30:
        start_merge_step = 30
    
    images = pipe(
        prompt=prompt,
        width=output_w,
        height=output_h,
        input_id_images=input_id_images,
        negative_prompt=negative_prompt,
        num_images_per_prompt=NUM_OUTPUTS,
        num_inference_steps=NUM_STEPS,
        start_merge_step=start_merge_step,
        generator=generator,
        guidance_scale=GUIDANCE_SCALE,
        id_embeds=id_embeds,
        image=sketch_image,
        adapter_conditioning_scale=adapter_scale,
        adapter_conditioning_factor=adapter_factor,
    ).images
    
    return images, seed


def main():
    print("=" * 50)
    print("PhotoMaker V2 CLI")
    print("=" * 50)
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    print(f"Using device: {device}")
    
    pipe = load_pipeline(device)
    face_detector = load_face_detector(device)
    
    try:
        images, used_seed = generate_image(pipe, face_detector, device)
        
        print(f"\nSaving {len(images)} image(s) to {output_dir}/")
        for i, img in enumerate(images):
            filename = f"output_seed{used_seed}_{i+1}.png"
            filepath = output_dir / filename
            img.save(filepath)
            print(f"  Saved: {filepath}")
        
        print(f"\nDone! Generated {len(images)} image(s) with seed {used_seed}")
        
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
