import gradio as gr
import os
import torch
import numpy as np
from PIL import Image
from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora
from torchvision import transforms

# Initialize the image processor
base_path = "black-forest-labs/FLUX.1-dev"    
lora_base_path = "./models"

# Ensure the model uses GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models and move to GPU
pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.float16).to(device)  # Use fp16 for reduced memory consumption
transformer = FluxTransformer2DModel.from_pretrained(base_path, subfolder="transformer", torch_dtype=torch.float16).to(device)
pipe.transformer = transformer

# Image transformation to tensor
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a tensor in [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1] for many models
])

# Function to clear the cache
def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()

# Image generation function
def generate_image(prompt, spatial_img, height, width, seed, control_type):
    # Load and process the spatial image
    spatial_img = spatial_img.convert("RGB")
    spatial_img = transform(spatial_img).unsqueeze(0).to(device)  # Convert to tensor and add batch dimension

    # Set the control type (e.g., Ghibli)
    if control_type == "Ghibli":
        lora_path = os.path.join(lora_base_path, "Ghibli.safetensors")
    set_single_lora(pipe.transformer, lora_path, lora_weights=[1], cond_size=512)

    # Prepare the generator for the specified seed and move it to the GPU
    generator = torch.Generator(device=device).manual_seed(seed)

    # Process the image
    spatial_imgs = [spatial_img] if spatial_img else []
    image = pipe(
        prompt,
        height=int(height),
        width=int(width),
        guidance_scale=3.5,
        num_inference_steps=25,
        max_sequence_length=512,
        generator=generator,  # Generator now uses the correct device (GPU)
        subject_images=[],
        spatial_images=spatial_imgs,
        cond_size=512,
    ).images[0]

    # Clear the cache after generation
    clear_cache(pipe.transformer)

    return image

# Gradio interface
def gradio_interface(prompt, spatial_img, height, width, seed, control_type):
    # Generate the image based on the user input
    generated_image = generate_image(prompt, spatial_img, height, width, seed, control_type)
    
    return generated_image

# Define the Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter a description for the generated image"),
        gr.Image(type="pil", label="Upload Spatial Image"),
        gr.Slider(128, 1024, step=1, label="Height", value=768),
        gr.Slider(128, 1024, step=1, label="Width", value=768),
        gr.Slider(1, 100, step=1, label="Seed", value=42),
        gr.Radio(["Ghibli", "Other"], label="Control Type", value="Ghibli")
    ],
    outputs=gr.Image(type="pil"),
    server_port=8080  # Run on port 8080
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=8080)
