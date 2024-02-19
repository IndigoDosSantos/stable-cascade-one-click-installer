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

def generate_images(prompt, height, width, negative_prompt, guidance_scale, num_inference_steps, num_images_per_prompt, generator):
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

        prompt = prompt.replace('\u00A0', ' ')  # Replace non-breaking spaces first

        # Replace excess spaces with single spaces
        space_pattern = r" +"
        prompt = re.sub(space_pattern, " ", prompt)

        # Handle commas: Replace multiple commas possibly separated by spaces with a single comma
        comma_pattern = r"(, )+"
        prompt = re.sub(comma_pattern, ",", prompt)

        # Remove any spaces before and after commas
        prompt = re.sub(r" ,", ",", prompt)
        prompt = re.sub(r", ", ",", prompt)

        # Remove any trailing or leading comma and spaces
        prompt = prompt.strip(", ")

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

# Styles list from https://latenightportrait.com/60-art-styles-explained-with-examples/#ib-toc-anchor-46
style_choices = {
    "Abstract Expressionism", "Academic Art", "Ancient Art", "Anti-Art", "Art Deco", "Art Nouveau", "Avant-garde", "Baroque", "Bauhaus", "Classicism", "CoBrA", "Color Field Painting", "Conceptual Art", "Constructivism", "Contemporary Art", "Cubism", "Dada / Dadism", "De Stijl", "Digital Art", "Dutch Golden Age", "Expressionism", "Fauvism", "Figurative", "Fluxus", "Folk Art", "Futurism", "Geometric", "Gothic Art", "Zero Group", "Harlem Renaisssance", "Hyperrealism", "Impressionism", "Installation Art", "Japonism", "Kinetic Art", "Land Art", "Magical Realism", "Minimalism", "Modern Art", "Naïve Art", "Nature Art", "Neoclassicism", "Neo-Impressionism", "Neo-Surrealism", "Neon Art", "Op Art", "Painterly", "Performance Art", "Photorealism", "Pointilism", "Pop Art", "Portraiture", "Post-Impressionism", "Postmodern Art", "Precisionism", "Primitivism", "Realism", "Renaissance Art", "Rococo", "Romanticism", "Spiritual Art", "Still Life", "Street Art", "Stuckism", "Suprematism", "Surrealism", "Symbolism", "Typography", "Ukiyo-e", "Urban"
}
technique = {
    "collage", "composition", "drawing", "etching", "fresco", "illustration", "mural", "painting", "photo", "portrait", "print", "representation", "sculpture", "sketch", "watercolor", "woodcut"
}
subject = {
    "animal", "architecture", "beach", "city", "cloud", "desert", "flower", "forest", "fruit", "furniture", "house", "insect", "island", "lake", "landscape", "leaf", "man", "moon", "mountain", "nonbinary", "object", "ocean", "person", "plant", "river", "rock", "star", "tree", "woman"
}
action = {
    "balancing", "blushing", "climbing", "concealing", "contemplating", "dancing", "daydreaming", "discovering", "embracing", "emerging", "entangling", "exploring", "floating", "gazing", "gliding", "hiding", "hovering", "laughing", "leaping", "listening", "meditating", "pondering", "reaching", "reflecting", "resisting", "searching", "transforming", "whispering", "wishing", "yearning"
}
affective_adverb = {
    "adoringly", "aggressively", "alarmedly", "amusedly", "angrily", "anxiously", "apathically", "approvingly", "arrogantly", "awkwardly", "bitterly", "blissfully", "boredly", "calmly", "compassionately", "confusedly", "contemptuously", "contently", "curiously", "cynically", "delightfully", "determinedly", "disapprovingly", "disbelievingly", "disgustedly", "dreamily", "eagerly", "enviously", "fearfully", "flirtingly", "fondly", "frustratedly", "gleefully", "gloomily", "gratefully", "happily", "hatefully", "hesitantly", "hopelessly", "hungrily", "inquisitively", "intensely", "joyfully", "longingly", "lovingly", "miserably", "mockingly", "nervously", "painfully", "patiently", "peacefully", "patiently", "peacefully", "playfully", "proudly", "questioningly", "regretfully", "sadly", "sarcastically", "skeptically", "worriedly", "yearningly"
}

def handle_dropdown_change(selected_option):
    # This function can handle changes in dropdown selections
    # For demonstration purposes, it doesn't do much, but you can extend it
    print(f"Dropdown selection changed: {selected_option}")
    return selected_option

with gr.Blocks() as demo:
    with gr.Column():
        prompt = gr.Textbox(label="Prompt")
        height = gr.Slider(minimum=512, maximum=2048, step=1, value=1024, label="Image Height")
        width = gr.Slider(minimum=512, maximum=2048, step=1, value=1024, label="Image Width")
        negative_prompt = gr.Textbox(label="Negative Prompt", value="")
        guidance_scale = gr.Slider(minimum=1, maximum=20, step=0.5, value=4.0, label="Guidance Scale")
        num_inference_steps = gr.Slider(minimum=1, maximum=150, step=1, value=30, label="Steps")
        num_images_per_prompt = gr.Number(label="Number of Images per Prompt", value=2)
        seed = gr.Number(label="Seed", value=-1)  # Removed the comma here

        generate_button = gr.Button("Generate Images")
        gallery = gr.Gallery(label="Generated Images")

        generate_button.click(
            fn=generate_images,
            inputs=[prompt, height, width, negative_prompt, guidance_scale, num_inference_steps, num_images_per_prompt, seed],
            outputs=[gallery]
        )

    with gr.Column():
        # Prompt Configurator dropdowns
        style_dropdown = gr.Dropdown(style_choices, label="Select a style")
        technique_dropdown = gr.Dropdown(technique, label="Select a technique")
        subject_dropdown = gr.Dropdown(subject, label="Select a subject")
        action_dropdown = gr.Dropdown(action, label="Select an action")
        affective_adverb_dropdown = gr.Dropdown(affective_adverb, label="Select an affective verb")
        # Removed the commas at the end of each line above

        # Assuming you want to do something with the dropdowns, like displaying the selected value
        output_text = gr.Textbox(label="Selected Option")
        style_dropdown.change(fn=handle_dropdown_change, inputs=[style_dropdown], outputs=[output_text])
        # Corrected the structure here, too

demo.launch(inbrowser=True)
