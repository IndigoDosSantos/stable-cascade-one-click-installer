# This file contains code that is derived from Stability AI's software products, 
# which are licensed under the Stability AI Non-Commercial Research Community License Agreement.
# Copyright (c) Stability AI Ltd. All Rights Reserved.
#
# The original work is provided by Stability AI and is available under the terms of the 
# Stability AI Non-Commercial Research Community License Agreement, dated November 28, 2023.
# For more information, see https://stability.ai/use-policy.

import gradio as gr
import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
import os
import uuid  # Import the uuid library

# Initialize the device and dtype
device = "cuda"
dtype = torch.bfloat16

# Preload models to avoid reloading them on each function call
prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=dtype).to(device)
decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", torch_dtype=dtype).to(device)

def generate_images(prompt, height, width, negative_prompt, guidance_scale, num_images_per_prompt):
    output_directory = "./Output"
    os.makedirs(output_directory, exist_ok=True)
    output_images = []

    with torch.cuda.amp.autocast(dtype=dtype):
        prior_output = prior(
            prompt=prompt,
            height=int(height),
            width=int(width),
            negative_prompt=negative_prompt,
            guidance_scale=float(guidance_scale),
            num_images_per_prompt=int(num_images_per_prompt),
        )
        decoder_output = decoder(
            image_embeddings=prior_output.image_embeddings,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=0.0,
            output_type="pil",
        ).images

    for image in decoder_output:
        unique_filename = f"generated_image_{uuid.uuid4()}.png"
        save_path = os.path.join(output_directory, unique_filename)
        image.save(save_path)
        output_images.append(save_path)
    
    return output_images

iface = gr.Interface(
    fn=generate_images,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Slider(minimum=512, maximum=2048, step=1, value=1024, label="Image Height"),
        gr.Slider(minimum=512, maximum=2048, step=1, value=1024, label="Image Width"),
        gr.Textbox(label="Negative Prompt", value=""),
        gr.Slider(minimum=1, maximum=20, step=0.5, value=7.5, label="Guidance Scale"),
        gr.Number(label="Number of Images per Prompt", value=2)
    ],
    outputs=gr.Gallery(label="Generated Images"),
    title="Image Generator",
    description="Generate images based on your prompts!"
)

iface.launch(inbrowser=True)
