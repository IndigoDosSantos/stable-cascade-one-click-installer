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
import re

# Initialize the device and dtype
device = "cuda"
dtype = torch.bfloat16

# Preload models to avoid reloading them on each function call
prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=dtype).to(device)
decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", torch_dtype=dtype).to(device)

def generate_images(prompt, height, width, negative_prompt, guidance_scale, num_inference_steps, num_images_per_prompt, generator, style_choices, copy_button):
    output_directory = "./Output"
    os.makedirs(output_directory, exist_ok=True)
    output_images = []
    calculated_steps_prior = int(num_inference_steps * 2 / 3)
    calculated_steps_decoder = int(num_inference_steps * 1 / 3)

    with torch.cuda.amp.autocast(dtype=dtype):
        if generator == -1: # Check if user wants a random seed
            generator = torch.Generator(device).manual_seed(torch.seed())
        else:
            generator = torch.Generator(device).manual_seed(generator)

        # Define the clean_prompt function
    def clean_prompt(prompt):
        """
        Cleans the prompt by removing 'a', 'the', excess spaces, and multiple commas, and handles spaces between commas.
        """
        # Remove 'a' and 'the' (case-insensitive)
        word_pattern = r"\b(a|the)\b"
        prompt = re.sub(word_pattern, "", prompt, flags=re.IGNORECASE)
        print("Prompt first step cleaning:", prompt)

        prompt = prompt.replace('\u00A0', ' ')  # Replace non-breaking spaces first

        # Replace excess spaces with single spaces
        space_pattern = r" +"
        prompt = re.sub(space_pattern, " ", prompt)
        print("Prompt second step cleaning:", prompt)

        # Handle commas: Replace multiple commas possibly separated by spaces with a single comma
        comma_pattern = r"(, )+"
        prompt = re.sub(comma_pattern, ",", prompt)
        print("Prompt third cleaning:", prompt)

        # Remove any spaces before and after commas
        prompt = re.sub(r" ,", ",", prompt)
        prompt = re.sub(r", ", ",", prompt)

        # Remove any trailing or leading comma and spaces
        prompt = prompt.strip(", ")
        print("Prompt fourth cleaning:", prompt)

        # Ensure there is exactly one space after each comma
        prompt = re.sub(r",", ", ", prompt)
        # Remove any possible double spaces that could have been introduced in the previous step
        prompt = re.sub(r" +", " ", prompt)

        return prompt


    # Sanitize user input prompt before using it
    cleaned_prompt = clean_prompt(prompt)
    print("Your prompt:", cleaned_prompt)
            
    prior_output = prior(
        prompt=cleaned_prompt,
        height=int(height),
        width=int(width),
        negative_prompt=negative_prompt,
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(calculated_steps_prior),
        num_images_per_prompt=int(num_images_per_prompt),
        generator=generator,
    )
    decoder_output = decoder(
        image_embeddings=prior_output.image_embeddings,
        prompt=cleaned_prompt,
        negative_prompt=negative_prompt,
        guidance_scale=0.0,
        num_inference_steps=calculated_steps_decoder,
        output_type="pil",
        generator=generator,
    ).images

    for image in decoder_output:
        unique_filename = f"generated_image_{uuid.uuid4()}.png"
        save_path = os.path.join(output_directory, unique_filename)
        image.save(save_path)
        output_images.append(save_path)
    
    return output_images

style_choices = {
    "Impressionism", "Surrealism", "Fauvism"
}
copy_button = gr.Button("Copy") # Create the copy button
dropdown = gr.Dropdown(style_choices, label="Select a style")

iface = gr.Interface(
    fn=generate_images,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Slider(minimum=512, maximum=2048, step=1, value=1024, label="Image Height"),
        gr.Slider(minimum=512, maximum=2048, step=1, value=1024, label="Image Width"),
        gr.Textbox(label="Negative Prompt", value=""),
        gr.Slider(minimum=1, maximum=20, step=0.5, value=4.0, label="Guidance Scale"), # CFG 4.0 is recommended for Stable Cascade
        gr.Slider(minimum=1, maximum=150, step=1, value=30, label="Steps"), # Is `step=1` necessary?
        gr.Number(label="Number of Images per Prompt", value=2),
        gr.Number(label="Seed", value=-1), # Need to add `value=-1` means random seed.
        gr.Dropdown(style_choices, label="Select a style"),
        gr.Button("Copy")
    ],
    outputs=gr.Gallery(label="Generated Images"),
    title="Image Generator",
    description="Generate images based on your prompts!"
)

iface.launch(inbrowser=True)
