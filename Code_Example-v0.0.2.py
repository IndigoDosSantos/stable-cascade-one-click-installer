import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
import os
import uuid  # Import the uuid library

# Set device and data type
device = "cuda"
dtype = torch.bfloat16

# User inputs for customization
prompt = input("Enter your prompt: ")
height = int(input("Enter the image height (e.g., 1024): "))
width = int(input("Enter the image width (e.g., 1024): "))
negative_prompt = input("Enter your negative prompt, if any (or press enter to skip): ")
guidance_scale = float(input("Enter the guidance scale (e.g., 4.0): "))
num_images_per_prompt = int(input("Enter the number of images per prompt (e.g., 2): "))

# Load models
prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=dtype).to(device)
decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", torch_dtype=dtype).to(device)

with torch.cuda.amp.autocast(dtype=dtype):
    prior_output = prior(
        prompt=prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
    )
    decoder_output = decoder(
        image_embeddings=prior_output.image_embeddings,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=0.0,
        output_type="pil",
    ).images

# Use a relative path for the output directory
output_directory = "./Output"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Display or save images
for i, image in enumerate(decoder_output):
    # Optional: Display the image
    # image.show()

    # Generate a unique filename using a UUID
    unique_filename = f"generated_image_{uuid.uuid4()}.png"
    save_path = os.path.join(output_directory, unique_filename)
    image.save(save_path)
    print(f"Saved: {save_path}")
