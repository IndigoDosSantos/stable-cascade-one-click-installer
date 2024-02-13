import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
import os
import uuid  # Import the uuid library

device = "cuda"
dtype = torch.bfloat16
num_images_per_prompt = 2

prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=dtype).to(device)
decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade",  torch_dtype=dtype).to(device)

prompt = ""
negative_prompt = ""

with torch.cuda.amp.autocast(dtype=dtype):
    prior_output = prior(
        prompt=prompt,
        height=1024,
        width=1024,
        negative_prompt=negative_prompt,
        guidance_scale=4.0,
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
output_directory = "./Output"  # This will create an 'Output' folder in the current working directory

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Display or save images
for i, image in enumerate(decoder_output):
    # Display the image (optional)
    image.show()

    # Generate a unique filename using a UUID
    unique_filename = f"generated_image_{uuid.uuid4()}.png"
    save_path = os.path.join(output_directory, unique_filename)
    image.save(save_path)
