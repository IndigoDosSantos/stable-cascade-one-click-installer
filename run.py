# This file contains code that is derived from Stability AI's software products, 
# which are licensed under the Stability AI Non-Commercial Research Community License Agreement.
# Copyright (c) Stability AI Ltd. All Rights Reserved.
#
# The original work is provided by Stability AI and is available under the terms of the 
# Stability AI Non-Commercial Research Community License Agreement, dated November 28, 2023.
# For more information, see https://stability.ai/use-policy.

from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
import gradio as gr
import json
import os
import re
import torch
import uuid

# Initialize global settings
device = "cuda"
dtype = torch.bfloat16
output_directory = "./Output"

def load_model(model_name):
    # Load model from disk every time it's needed
    if model_name == "prior":
        model = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=dtype).to(device)
    elif model_name == "decoder":
        model = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", torch_dtype=dtype).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

def clean_prompt(prompt):
    """
    Cleans and formats the prompt text.
    
    Removes unnecessary articles ('a' and 'the'), standardizes whitespace and comma usage,
    and ensures the sanitized prompt has the format "text, text, text text".
    """
    prompt = re.sub(r"\b(a|the)\b", "", prompt, flags=re.IGNORECASE)
    prompt = re.sub(r"\s+", " ", prompt).strip()
    prompt = re.sub(r"\s*,\s*", ", ", prompt)
    prompt = prompt.strip(',')
    prompt_parts = [part.strip() for part in prompt.split(',')]
    prompt_parts = [part for part in prompt_parts if part]
    prompt = ', '.join(prompt_parts)
    return prompt

def generate_images(prompt, height, width, negative_prompt, guidance_scale, num_inference_steps, num_images_per_prompt, seed):
    """
    Generates images based on the provided parameters and settings.
    """
    os.makedirs(output_directory, exist_ok=True)
    output_images = []
    calculated_steps_prior = int(num_inference_steps * 2 / 3)
    calculated_steps_decoder = int(num_inference_steps * 1 / 3)

    # Sanitize user input prompt before using it
    cleaned_prompt = clean_prompt(prompt)
    print("Processed prompt:", cleaned_prompt)

    # Load, use, and discard the prior model
    prior = load_model("prior")
    with torch.cuda.amp.autocast(dtype=dtype): 
        generator = torch.Generator(device).manual_seed(torch.seed() if seed == -1 else seed)

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
    del prior  # Explicitly delete the model to help with memory management
    torch.cuda.empty_cache()  # Clear the CUDA cache to free up unused memory

    # Load, use, and discard the decoder model
    decoder = load_model("decoder")
    decoder_output = decoder(
        image_embeddings=prior_output.image_embeddings,
        prompt=cleaned_prompt,
        negative_prompt=negative_prompt,
        guidance_scale=0.0,
        num_inference_steps=calculated_steps_decoder,
        output_type="pil",
        generator=generator,
    ).images
    del decoder  # Explicitly delete the model to help with memory management
    torch.cuda.empty_cache()  # Clear the CUDA cache to free up unused memory

    for image in decoder_output:
        unique_filename = f"generated_image_{uuid.uuid4()}.png"
        save_path = os.path.join(output_directory, unique_filename)
        image.save(save_path)
        output_images.append(save_path)
    
    return output_images

# Load the JSON data
with open('prompt_configurator/data_prompt_configurator.json', 'r') as file:
    data = json.load(file)

# Retrieves style_choices, technique, subject, etc., from the loaded JSON file
# Styles list from https://latenightportrait.com/60-art-styles-explained-with-examples/#ib-toc-anchor-46
style_choices = data['style_choices']
technique = data['technique']
subject = data['subject']
action = data['action']
affective_adverb = data['affective_adverb']
physique = data['physique']
hairstyle = data['hairstyle']
facial_features = data['facial_features']
top = data['top']
bottom = data['bottom']
background = data['background']
lighting = data['lighting']
color = data['color']
texture = data['texture']
camera = data['camera']
framing = data['framing']
mood = data['mood']
story = data['story']
post_processing = data['post_processing']

# UI Layout putting Configurator blocks inside a function for clarity.
def configure_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as demo: # Change to your desired theme
        gr.HTML("""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed&display=swap" rel="stylesheet">
        """)

        gr.Markdown("# Stable Cascade Image Generator")

        # CSS placement
        gr.HTML("""
            <style>
                .my-slider-container {
                    height: auto;
                }
            </style>
        """)
        with gr.Column():
            gallery = gr.Gallery(label="Generated Images")
            generate_button = gr.Button("Generate")
        with gr.Row():
            prompt = gr.Textbox(label="Prompt")
            negative_prompt = gr.Textbox(label="Negative Prompt", value="")

        with gr.Row(): # For three parameter columns
            with gr.Column():
                # components in left column
                width = gr.Slider(minimum=512, maximum=2048, step=1, value=1024, label="Image Width")
                height = gr.Slider(minimum=512, maximum=2048, step=1, value=1024, label="Image Height")
            with gr.Column():
                # components in central column
                num_inference_steps = gr.Slider(minimum=1, maximum=150, step=1, value=30, label="Steps")
                num_images_per_prompt = gr.Number(label="Number of Images per Prompt", value=2)
            with gr.Column():
                # components in right column
                guidance_scale = gr.Slider(minimum=1, maximum=20, step=0.5, value=4.0, label="Guidance Scale")
                seed = gr.Number(label="Seed", value=-1)

            generate_button.click(
                fn=generate_images,
                inputs=[prompt, height, width, negative_prompt, guidance_scale, num_inference_steps, num_images_per_prompt, seed],
                outputs=[gallery]
            )

        def handle_dropdown_change(*args):
            selected_options = ' '.join([str(arg) for arg in args if arg])
            return selected_options
            
        configurator_group = gr.Group(visible=True) # Group to hold the configurator elements. Initially hidden.
            
        with configurator_group:
            with gr.Row():
                # Prompt Configurator dropdowns
                output_text = gr.Textbox("Your configured prompt.", label="Selected Option")
            with gr.Row():
                style_dropdown = gr.Dropdown(style_choices, label="Style")
                technique_dropdown = gr.Dropdown(technique, label="Technique")
                subject_dropdown = gr.Dropdown(subject, label="Subject")
            with gr.Row():
                action_dropdown = gr.Dropdown(action, label="Action")
                affective_adverb_dropdown = gr.Dropdown(affective_adverb, label="Affective verb")
            with gr.Row():
                physique_dropdown = gr.Dropdown(physique, label="Physique")
                hairstyle_dropdown = gr.Dropdown(hairstyle, label="Hairstyle")
                facial_features_dropdown = gr.Dropdown(facial_features, label="Facial features")
                top_dropdown = gr.Dropdown(top, label="Top")
                bottom_dropdown = gr.Dropdown(bottom, label="Bottom")
            with gr.Row():
                background_dropdown = gr.Dropdown(background, label="Background")
                lighting_dropdown = gr.Dropdown(lighting, label="Lighting")
            with gr.Row():
                color_dropdown = gr.Dropdown(color, label="Color")
                texture_dropdown = gr.Dropdown(texture, label="Texture")
            with gr.Row():
                camera_dropdown = gr.Dropdown(camera, label="Camera")
                framing_dropdown = gr.Dropdown(framing, label="Framing")
            with gr.Row():
                mood_dropdown = gr.Dropdown(mood, label="Mood")
                story_dropdown = gr.Dropdown(story, label="Story")
            with gr.Row():
                post_processing_dropdown = gr.Dropdown(post_processing, label="Post-processing")

                    # Assuming you want to do something with the dropdowns, like displaying the selected value
                style_dropdown.change(
                    fn=handle_dropdown_change,
                    inputs=[style_dropdown, technique_dropdown, subject_dropdown, action_dropdown, affective_adverb_dropdown, physique_dropdown, hairstyle_dropdown, facial_features_dropdown, top_dropdown, bottom_dropdown, background_dropdown, lighting_dropdown, color_dropdown, texture_dropdown, camera_dropdown,  framing_dropdown, mood_dropdown, story_dropdown, post_processing_dropdown],
                    outputs=[output_text]
                )
                technique_dropdown.change(
                    fn=handle_dropdown_change,
                    inputs=[style_dropdown, technique_dropdown, subject_dropdown, action_dropdown, affective_adverb_dropdown, physique_dropdown, hairstyle_dropdown, facial_features_dropdown, top_dropdown, bottom_dropdown, background_dropdown, lighting_dropdown, color_dropdown, texture_dropdown, camera_dropdown,  framing_dropdown, mood_dropdown, story_dropdown, post_processing_dropdown],
                    outputs=[output_text]
                )
                subject_dropdown.change(
                    fn=handle_dropdown_change,
                    inputs=[style_dropdown, technique_dropdown, subject_dropdown, action_dropdown, affective_adverb_dropdown, physique_dropdown, hairstyle_dropdown, facial_features_dropdown, top_dropdown, bottom_dropdown, background_dropdown, lighting_dropdown, color_dropdown, texture_dropdown, camera_dropdown,  framing_dropdown, mood_dropdown, story_dropdown, post_processing_dropdown],
                    outputs=[output_text]
                )
                action_dropdown.change(
                    fn=handle_dropdown_change,
                    inputs=[style_dropdown, technique_dropdown,  subject_dropdown, action_dropdown, affective_adverb_dropdown, physique_dropdown, hairstyle_dropdown, facial_features_dropdown, top_dropdown, bottom_dropdown, background_dropdown, lighting_dropdown, color_dropdown, texture_dropdown, camera_dropdown,  framing_dropdown, mood_dropdown, story_dropdown, post_processing_dropdown],
                    outputs=[output_text]
                )
                affective_adverb_dropdown.change(
                    fn=handle_dropdown_change,
                    inputs=[style_dropdown, technique_dropdown, subject_dropdown, action_dropdown, affective_adverb_dropdown, physique_dropdown, hairstyle_dropdown, facial_features_dropdown, top_dropdown, bottom_dropdown, background_dropdown, lighting_dropdown, color_dropdown, texture_dropdown, camera_dropdown,  framing_dropdown, mood_dropdown, story_dropdown, post_processing_dropdown],
                    outputs=[output_text]
                )
                physique_dropdown.change(
                    fn=handle_dropdown_change,
                    inputs=[style_dropdown, technique_dropdown, subject_dropdown, action_dropdown, affective_adverb_dropdown, physique_dropdown, hairstyle_dropdown, facial_features_dropdown, top_dropdown, bottom_dropdown, background_dropdown, lighting_dropdown, color_dropdown, texture_dropdown, camera_dropdown,  framing_dropdown, mood_dropdown, story_dropdown, post_processing_dropdown],
                    outputs=[output_text]
                )
                hairstyle_dropdown.change(
                    fn=handle_dropdown_change,
                    inputs=[style_dropdown, technique_dropdown, subject_dropdown, action_dropdown, affective_adverb_dropdown, physique_dropdown, hairstyle_dropdown, facial_features_dropdown, top_dropdown, bottom_dropdown, background_dropdown, lighting_dropdown, color_dropdown, texture_dropdown, camera_dropdown,  framing_dropdown, mood_dropdown, story_dropdown, post_processing_dropdown],
                    outputs=[output_text]
                )
                facial_features_dropdown.change(
                    fn=handle_dropdown_change,
                    inputs=[style_dropdown, technique_dropdown, subject_dropdown, action_dropdown, affective_adverb_dropdown, physique_dropdown, hairstyle_dropdown, facial_features_dropdown, top_dropdown, bottom_dropdown, background_dropdown, lighting_dropdown, color_dropdown, texture_dropdown, camera_dropdown,  framing_dropdown, mood_dropdown, story_dropdown, post_processing_dropdown],
                    outputs=[output_text]
                )
                top_dropdown.change(
                    fn=handle_dropdown_change,
                    inputs=[style_dropdown, technique_dropdown, subject_dropdown, action_dropdown, affective_adverb_dropdown, physique_dropdown, hairstyle_dropdown, facial_features_dropdown, top_dropdown, bottom_dropdown, background_dropdown, lighting_dropdown, color_dropdown, texture_dropdown, camera_dropdown,  framing_dropdown, mood_dropdown, story_dropdown, post_processing_dropdown],
                    outputs=[output_text]
                )
                bottom_dropdown.change(
                    fn=handle_dropdown_change,
                    inputs=[style_dropdown, technique_dropdown, subject_dropdown, action_dropdown, affective_adverb_dropdown, physique_dropdown, hairstyle_dropdown, facial_features_dropdown, top_dropdown, bottom_dropdown, background_dropdown, lighting_dropdown, color_dropdown, texture_dropdown, camera_dropdown,  framing_dropdown, mood_dropdown, story_dropdown, post_processing_dropdown],
                    outputs=[output_text]
                )
                background_dropdown.change(
                    fn=handle_dropdown_change,
                    inputs=[style_dropdown, technique_dropdown, subject_dropdown, action_dropdown, affective_adverb_dropdown, physique_dropdown, hairstyle_dropdown, facial_features_dropdown, top_dropdown, bottom_dropdown, background_dropdown, lighting_dropdown, color_dropdown, texture_dropdown, camera_dropdown,  framing_dropdown, mood_dropdown, story_dropdown, post_processing_dropdown],
                    outputs=[output_text]
                )
                lighting_dropdown.change(
                    fn=handle_dropdown_change,
                    inputs=[style_dropdown, technique_dropdown, subject_dropdown, action_dropdown, affective_adverb_dropdown, physique_dropdown, hairstyle_dropdown, facial_features_dropdown, top_dropdown, bottom_dropdown, background_dropdown, lighting_dropdown, color_dropdown, texture_dropdown, camera_dropdown,  framing_dropdown, mood_dropdown, story_dropdown, post_processing_dropdown],
                    outputs=[output_text]
                )
                color_dropdown.change(
                    fn=handle_dropdown_change,
                    inputs=[style_dropdown, technique_dropdown, subject_dropdown, action_dropdown, affective_adverb_dropdown, physique_dropdown, hairstyle_dropdown, facial_features_dropdown, top_dropdown, bottom_dropdown, background_dropdown, lighting_dropdown, color_dropdown, texture_dropdown, camera_dropdown,  framing_dropdown, mood_dropdown, story_dropdown, post_processing_dropdown],
                    outputs=[output_text]
                )
                texture_dropdown.change(
                    fn=handle_dropdown_change,
                    inputs=[style_dropdown, technique_dropdown, subject_dropdown, action_dropdown, affective_adverb_dropdown, physique_dropdown, hairstyle_dropdown, facial_features_dropdown, top_dropdown, bottom_dropdown, background_dropdown, lighting_dropdown, color_dropdown, texture_dropdown, camera_dropdown,  framing_dropdown, mood_dropdown, story_dropdown, post_processing_dropdown],
                    outputs=[output_text]
                )
                camera_dropdown.change(
                    fn=handle_dropdown_change,
                    inputs=[style_dropdown, technique_dropdown, subject_dropdown, action_dropdown, affective_adverb_dropdown, physique_dropdown, hairstyle_dropdown, facial_features_dropdown, top_dropdown, bottom_dropdown, background_dropdown, lighting_dropdown, color_dropdown, texture_dropdown, camera_dropdown,  framing_dropdown, mood_dropdown, story_dropdown, post_processing_dropdown],
                    outputs=[output_text]
                )
                framing_dropdown.change(
                    fn=handle_dropdown_change,
                    inputs=[style_dropdown, technique_dropdown, subject_dropdown, action_dropdown, affective_adverb_dropdown, physique_dropdown, hairstyle_dropdown, facial_features_dropdown, top_dropdown, bottom_dropdown, background_dropdown, lighting_dropdown, color_dropdown, texture_dropdown, camera_dropdown,  framing_dropdown, mood_dropdown, story_dropdown, post_processing_dropdown],
                    outputs=[output_text]
                )
                mood_dropdown.change(
                    fn=handle_dropdown_change,
                    inputs=[style_dropdown, technique_dropdown, subject_dropdown, action_dropdown, affective_adverb_dropdown, physique_dropdown, hairstyle_dropdown, facial_features_dropdown, top_dropdown, bottom_dropdown, background_dropdown, lighting_dropdown, color_dropdown, texture_dropdown, camera_dropdown,  framing_dropdown, mood_dropdown, story_dropdown, post_processing_dropdown],
                    outputs=[output_text]
                )
                story_dropdown.change(
                    fn=handle_dropdown_change,
                    inputs=[style_dropdown, technique_dropdown, subject_dropdown, action_dropdown, affective_adverb_dropdown, physique_dropdown, hairstyle_dropdown, facial_features_dropdown, top_dropdown, bottom_dropdown, background_dropdown, lighting_dropdown, color_dropdown, texture_dropdown, camera_dropdown,  framing_dropdown, mood_dropdown, story_dropdown, post_processing_dropdown],
                    outputs=[output_text]
                )
                post_processing_dropdown.change(
                    fn=handle_dropdown_change,
                    inputs=[style_dropdown, technique_dropdown, subject_dropdown, action_dropdown, affective_adverb_dropdown, physique_dropdown, hairstyle_dropdown, facial_features_dropdown, top_dropdown, bottom_dropdown, background_dropdown, lighting_dropdown, color_dropdown, texture_dropdown, camera_dropdown,  framing_dropdown, mood_dropdown, story_dropdown, post_processing_dropdown],
                    outputs=[output_text]
                )
    return demo  # Return the Blocks object for external access

# Adjusted call to configure_ui and launching
demo_ui = configure_ui()  # This will receive the 'demo' object returned from the function
demo_ui.launch(inbrowser=True)  # Use the returned object to launch the UI
