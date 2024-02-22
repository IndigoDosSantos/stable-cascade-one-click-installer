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
import uuid
import re

# Initialize the device and dtype
device = "cuda"
dtype = torch.bfloat16

def load_model(model_name):
    # Load model from disk every time it's needed
    if model_name == "prior":
        model = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=dtype).to(device)
    elif model_name == "decoder":
        model = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", torch_dtype=dtype).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

def generate_images(prompt, height, width, negative_prompt, guidance_scale, num_inference_steps, num_images_per_prompt, generator):
    output_directory = "./Output"
    os.makedirs(output_directory, exist_ok=True)
    output_images = []
    calculated_steps_prior = int(num_inference_steps * 2 / 3)
    calculated_steps_decoder = int(num_inference_steps * 1 / 3)

    # Load, use, and discard the prior model
    prior = load_model("prior")
    with torch.cuda.amp.autocast(dtype=dtype): 
        generator = torch.Generator(device).manual_seed(torch.seed() if generator == -1 else generator)

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

# Styles list from https://latenightportrait.com/60-art-styles-explained-with-examples/#ib-toc-anchor-46
style_choices = {
    "Abstract Expressionism", "Academic Art", "Ancient Art", "Anti-Art", "Art Deco", "Art Nouveau", "Avant-garde", "Baroque", "Bauhaus", "Classicism", "CoBrA", "Color Field Painting", "Conceptual Art", "Constructivism", "Contemporary Art", "Cubism", "Dada", "De Stijl", "Digital Art", "Dutch Golden Age", "Expressionism", "Fauvism", "Figurative", "Fluxus", "Folk Art", "Futurism", "Geometric", "Gothic Art", "Zero Group", "Harlem Renaisssance", "Hyperrealism", "Impressionism", "Installation Art", "Japonism", "Kinetic Art", "Land Art", "Magical Realism", "Minimalism", "Modern Art", "Naïve Art", "Nature Art", "Neoclassicism", "Neo-Impressionism", "Neo-Surrealism", "Neon Art", "Op Art", "Painterly", "Performance Art", "Photorealism", "Pointilism", "Pop Art", "Portraiture", "Post-Impressionism", "Postmodern Art", "Precisionism", "Primitivism", "Realism", "Renaissance Art", "Rococo", "Romanticism", "Spiritual Art", "Still Life", "Street Art", "Stuckism", "Suprematism", "Surrealism", "Symbolism", "Typography", "Ukiyo-e", "Urban"
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
physique = {
    "angular", "athletic", "brawny", "chubby", "curvaceous", "delicate", "elongated", "emaciated", "flabby", "fleshy", "frail", "gaunt", "graceful", "heavyset", "herculean", "lanky", "lean", "lithe", "plump", "robust", "sculpted", "sinewy", "slender", "statuesque", "stocky", "supple", "toned", "voluptuous", "well-developed", "well-proportioned", "wiry"
}
hairstyle = {
    "afro", "asymmetrical cut", "baby bangs", "beehive", "bob cut", "bowl cut", "box braids", "braided crown", "braided updo", "buzz cut", "caesar cut", "cornrows", "crew cut", "crimped hair", "cropped cut", "curly undercut", "curtain bangs", "disheveled bob", "dreadlocks", "dutch braids", "dyed tips", "elegant chignon", "extensions", "fade", "faux hawk", "feathered layers", "finger waves", "fishtail braid", "french braid", "french twist", "half-up half-down", "high ponytail", "high top fade", "hime cut", "highlights", "layered cut", "locs", "long and sleek", "long side bangs", "low bun", "low ponytail", "messy bun", "middle part", "mohawk", "mullet", "ombre", "pageboy cut", "permed", "pixie cut", "pompadour", "quiff", "shag haircut", "shaved head", "side part", "slicked back", "space buns", "textured weaves", "top knot", "twists", "undone weaves"
}
facial_features = {
    "almond-shaped eyes", "arched eyebrows", "beauty mark", "bushy eyebrows", "high cheekbones", "chiseled jawline", "cleft chin", "crooked nose", "cupid's bow lips", "dimples", "downturned eyes", "expressive eyes", "freckles", "full lips", "high forehead", "hooded eyes", "long eyelashes", "piercing eyes", "pointed chin", "prominent nose", "rosy cheeks", "round face", "scars", "sharp features", "soft features", "square jawline", "strong brow line", "thin lips", "upturned nose", "wide-set eyes"
}
top = {
    "armored breastplate", "asymmetrical top", "beaded necklace", "blazer", "blouse", "bolero jacket", "bowling shirt", "brooch", "button-down shirt", "camisole", "cardigan", "cloak", "corset", "cowl neck sweater", "cravat", "crop top", "denim jacket", "dress shirt", "flowing scarf", "formal vest", "graphic t-shirt", "halter top", "henley shirt", "high-neck blouse", "hoodie", "infinity scarf", "jacket", "jean jacket", "jersey", "kimono", "knit shawl", "lace top", "leather jacket", "military jacket", "necktie", "off-the-shoulder top", "one-shoulder top", "oversized sweater", "oxford shirt", "pashmina", "peasant blouse", "pendant necklace", "polo shirt", "poncho", "puffy sleeves", "robe", "scarf", "shawl", "sheer top", "shoulder pads", "silk scarf", "spaghetti strap top", "suit jacket", "sundress", "sweater", "tank top", "turtleneck", "tuxedo jacket", "vest", "vintage tee"
}
bottom = {
    "baggy trousers", "ballet flats", "bell-bottom jeans", "belt", "bikini bottom", "board shorts", "ankle boots", "knee-high boots", "cowboy boots", "boots", "cargo pants", "chaps", "combat boots", "corduroy pants", "culottes", "denims shorts", "dress pants", "fishnet stockings", "flared pants", "flip-flops", "formal trousers", "garter belt", "gladiator sandals", "harem pants", "high heels", "high-waisted jeans", "jodhpurs", "kilt", "knee-high socks", "leather pants", "leg warmers", "leggings", "linen pants", "loafers", "long skirt", "maxi dress", "midi skirt", "miniskirt", "moccasins", "overall", "palazzo pants", "pantsuit", "pencil skirt", "platform shoes", "pleated skirt", "ripped jeans", "sandals", "sarong", "satin skirt", "silk pajama", "skinny jeans", "slacks", "sneakers", "socks", "ankle socks", "crew socks", "dress socks", "stilettos", "stockings", "sweatpants", "swim trunks", "tights", "track pants", "tuxedo pants", "wide-leg pants", "workout shorts", "yoga pants"
}
background = {
    "abstract cityscape", "ancient ruins", "barren desert", "blooming meadow", "blurred bokeh lights", "bustling city street", "calm ocean horizon", "cobblestone alleyway", "colorful coral reef", "cozy living room", "crackling firespace", "dense forest", "dramatic cloudspace", "empty warehouse", "foggy mountaintop", "futuristic cityscape", "glowing sunset", "graffiti-covered wall", "grand palace interior", "grassy field", "historic battlefield", "idyllic beach", "industrial factory", "laboratory interior", "lush rainforest", "milky way galaxy", "minimalist white backdrop", "modern art gallery", "moonlit landscape", "mossy forest floor", "nostalgic countryside", "old-fashioned library", "ornate cathedral", "pastel-colored sky", "peeling paint texture", "rain-soaked window", "rolling hills", "rustic farmhouse", "sandy beach", "school classroom", "shaded forest path", "shadowy silhouettes", "snowy mountain peak", "sparkling city lights", "stark desert landscape", "starry night sky", "stormy seascape", "sunlight filtering through leaves", "tranquil lake", "tropical island", "urban rooftops", "vibrant flower garden", "weathered barn", "weathered stone wall", "whimsical fairytail forest", "windswept cliffside", "wooden dock at sunset", "workshop filled with tools", "wrought iron fence", "yellow brick road"
}
lighting = {
    "backlighting", "bright sunlight", "candlelight", "catchlights", "chiaroscuro", "cool lighting", "dappled light", "daylight balanced light", "diffused light", "dim lighting", "dramatic shadows", "dusk lighting", "even lighting", "firelight", "flat lighting", "floodlight", "fluorescent lighting", "foggy light", "footlights", "frontal lighting", "golden hour light", "hard light", "harsh shadows", "hazy light", "high contrast lighting", "highlight", "indoor lighting", "intense light", "lamplight", "lens flares", "low contrast lighting", "low-key lighting", "misty light", "modeling light", "mottled light", "muted light", "mysterious shadows", "natural light", "neon lights", "noir lighting", "outdoor lighting", "overcast light", "Rembrandt lighting", "rim light", "ring light", "side lighting", "silhouetted figure", "soft light", "softbox lighting", "speckled light", "spot light", "starlight", "stage lighting", "sunlight through window", "sunrise light", "sunset light", "three-point lighting", "tungsten lighting", "warm lighting"
}
color = {
    "analogous colors", "bold color blocking", "complementary colors", "cool color palette", "desaturated colors", "earthy tones", "faded colors", "gradient", "harmonious color scheme", "high-contrast colors", "limited color palette", "metallic colors", "monochromatic color scheme", "muted colors", "neon colors", "ombre effect", "pastel colors", "primary colors", "rich jewel tones", "selective color", "split-complementary colors", "triadic color scheme", "unconventional color combinatiosn", "vibrant colors", "vintage color palette", "warm color palette", "warm vs. cool contrast", "watercolor effect"
}
texture = {
    "bumpy texture", "coarse texture", "cracked surface", "crinkled texture", "delicate texture", "fabric texture", "flaky texture", "furry texture", "glossy surface", "grain of wood", "gritty texture", "hard surface", "leahter texture", "marble texture", "matte finish", "metallic texture", "organic texture", "patterned texture", "polished surface", "reflective surface", "rough texture", "scaly texture", "shiny surface", "smooth texture", "soft texture", "stone texture", "textured brushstrokes", "weathered surface", "woven texture"
}
camera = {
    "action camera", "analog camera", "bridge camera", "cinema camera", "compact camera", "DSLR camera", "film camera", "instant camera", "large format camera", "medium format camera", "mirrorless camera", "point-and-shoot camera", "rangefinder camera", "toy camera", "twin-lens reflex TLR camera", "aperture priority mode", "autofocus", "black and white mode", "bokeh effect", "burst mode", "deep depth of field", "exposure compensation", "fast shutter speed", "fisheye lens", "flash photography", "grain film photography", "high ISO", "long exposure", "low ISO", "macro lens", "manual focus", "manual mode", "narrow aperture", "f/16", "f/22", "night photography mode", "noise reduction", "panoramic mode", "portrait mode", "shallow depth of field", "shutter priority mode", "telephoto lens", "time-lapse photography", "tilt-shift lens", "underexposed", "overexposed", "vibrant colors", "vignetting effect", "wide aperture", "f/1.8", "f/2.8", "wide-angle lens", "zoom lens"
}
framing = {
    "aerial perspective", "asymmetry", "balance", "bird's eye view", "centered composition", "close-up", "dead space", "negative space", "deep focus", "depth of field", "diagonal lines", "diagonally split composition", "dutch angle", "tilted angle", "extreme close-up", "foreground", "framing with", "geometric shapes", "golden ratio", "grid composition", "headroom", "heavy vignetting", "high angle shot", "horizontal lines", "intentional cropping", "isolation of subject", "juxtaposition of elements", "leading lines", "long shot", "look space", "space in front of gaze", "medium shot", "minimalist composition", "off-center subject", "off-kilter framing", "organic shapes", "overlapping elements", "panoramic view", "pattern interruption", "perspective", "one-point perspective", "two-point-perspective", "three-point perspective", "placement of subject", "positive space", "reflections", "repetition", "rule of odds", "rule of thirds", "scale and proportion", "selective focus", "shallow depth of field", "silhouettes", "tight framing", "triangular composition", "tunnel vision effect", "unusual perspective", "vertical lines", "worm's eye view", "wide shot"
}
mood = {
    "air of mistery", "brooding atmosphere", "calm and tranquil", "cheerful and bright", "claustrophobic feel", "contemplative mood", "dark and ominous", "dreamlike quality", "dramatic atmosphere", "eerie feeling", "energetic and lively", "enigmatic feeling", "ethereal atmosphere", "evocative of nostalgia", "expansive feeling", "gloomy and somber", "grim atmosphere", "haunting beauty", "heavy atmosphere", "hopeful and optimistic", "humorous and playful", "intense mood", "intimate feeling", "introspective mood", "isolation and loneliness", "joyful and celebratory", "light and airy", "lighthearted and whimsical", "love and tenderness", "melancholy and longing", "menacing and threatening", "moody and introspective", "nostaligic atmosphere", "ominous atmosphere", "peaceful and seren", "pensive atmosphere", "playful and carefree", "powerful and impactful", "quiet solitude", "reflective mood", "relaxed and contented", "romantic atmosphere", "sense of awe", "sense of foreboding", "sense of wonder", "somber and sorrowful", "spiritual feeling", "sublime awe", "surreal and dreamlike", "tension and unease", "triumphant and victorious", "twisted humor", "unsettling feeling", "uplifting and inspiring", "vintage feel", "whimsical and playful", "wistful and longing", "wonder and amazement"
}
story = {
    "call to action", "fleeting moment", "glimpse of the past", "journey of discovery", "moment of contemplation", "moment of triumph", "new beginning", "playful interaction", "sense of belonging", "sense of foreboding", "shared secret", "struggle for survival", "tale of friendship", "twist of fate", "warning", "prophecy", "act of rebellion", "enduring mystery", "expression of grief", "unexpected encounter", "anticipation of change", "beauty in mundane", "celebration of life", "consequences of actions", "cycles of nature", "dreams and aspirations", "exploration of identity", "facing inner demons", "finding strength in adversity", "fleeting beauty", "hidden desires", "holding on to hope", "humility in the face of nature", "inner turmoil", "intimate connection", "isolation and loneliness", "joy of simple things", "legacy and impact", "metamorphosis", "transformation", "power of imagination", "precious memories", "pursuit of hapiness", "quest for knowledge", "reconciliation and forgiveness", "reflection on mortality", "ritual", "tradition", "struggle for power", "supernatural phenomena", "symbolism", "allegory", "cost of ambition", "human spirit", "passage of time", "price of progress", "search for meaning", "unexpected path", "weight of responsibility", "wilderness within", "traces of existence", "transcending boundaries", "triumph over adversity", "unconditional love"
}
post_processing = {
    "black and white conversion", "brightness adjustment", "color correction", "color grading", "contrast adjustment", "depth of field manipulation", "digital painting techniques", "dodging", "burning", "exposure adjustment", "film grain simulation", "filter effects", "gradient overlay", "HDR creation", "healing brush", "high-pass sharpening", "lens correction", "levels adjustment", "noise reduction", "object removal", "perspective correction", "retouching", "selective color adjustments", "sharpening", "skin smoothing", "special effects", "light leaks", "split toning", "vignette addition", "white balance adjustment"
}

def handle_dropdown_change(*args):
    selected_options = ' '.join([str(arg) for arg in args if arg])
    return selected_options

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
            gr.HTML("""<div class="my-slider-container">""")
            width = gr.Slider(minimum=512, maximum=2048, step=1, value=1024, label="Image Width")
            height = gr.Slider(minimum=512, maximum=2048, step=1, value=1024, label="Image Height")
            gr.HTML("""</div>""")
        with gr.Column():
            # components in central column
            gr.HTML("""<div class="my-slider-container">""")
            num_inference_steps = gr.Slider(minimum=1, maximum=150, step=1, value=30, label="Steps")
            num_images_per_prompt = gr.Number(label="Number of Images per Prompt", value=2)
            gr.HTML("""</div>""")
        with gr.Column():
            # components in right column
            gr.HTML("""<div class="my-slider-container">""")
            guidance_scale = gr.Slider(minimum=1, maximum=20, step=0.5, value=4.0, label="Guidance Scale")
            seed = gr.Number(label="Seed", value=-1)
            gr.HTML("""</div>""")

        generate_button.click(
            fn=generate_images,
            inputs=[prompt, height, width, negative_prompt, guidance_scale, num_inference_steps, num_images_per_prompt, seed],
            outputs=[gallery]
        )
        
    with gr.Blocks():
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
            inputs=[style_dropdown, technique_dropdown, subject_dropdown, action_dropdown, affective_adverb_dropdown],
            outputs=[output_text]
        )
        subject_dropdown.change(
            fn=handle_dropdown_change,
            inputs=[style_dropdown, technique_dropdown, subject_dropdown, action_dropdown, affective_adverb_dropdown],
            outputs=[output_text]
        )
        action_dropdown.change(
            fn=handle_dropdown_change,
            inputs=[style_dropdown, technique_dropdown,  subject_dropdown, action_dropdown, affective_adverb_dropdown],
            outputs=[output_text]
        )
        affective_adverb_dropdown.change(
            fn=handle_dropdown_change,
            inputs=[style_dropdown, technique_dropdown,  subject_dropdown, action_dropdown, affective_adverb_dropdown],
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

demo.launch(inbrowser=True)
