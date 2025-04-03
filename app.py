import os
import json
import torch
import io
import base64
from PIL import Image
from flask import Flask, request, jsonify
from safetensors.torch import save_file
from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora

# Initialize the image processor
base_path = "black-forest-labs/FLUX.1-dev"    
lora_base_path = "./models"

pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16)
transformer = FluxTransformer2DModel.from_pretrained(base_path, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe.transformer = transformer
pipe.to("cuda")

def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()

# Initialize Flask app
app = Flask(__name__)

# Helper function to generate the image
def single_condition_generate_image(prompt, spatial_img, height, width, seed, control_type):
    # Set the control type (e.g., "Ghibli")
    if control_type == "Ghibli":
        lora_path = os.path.join(lora_base_path, "Ghibli.safetensors")
        set_single_lora(pipe.transformer, lora_path, lora_weights=[1], cond_size=512)
    
    # Process the image
    spatial_imgs = [spatial_img] if spatial_img else []
    image = pipe(
        prompt,
        height=int(height),
        width=int(width),
        guidance_scale=3.5,
        num_inference_steps=25,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(seed), 
        subject_images=[],
        spatial_images=spatial_imgs,
        cond_size=512,
    ).images[0]
    clear_cache(pipe.transformer)
    
    # Convert PIL image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return img_base64

# API endpoint for image generation
@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        # Get JSON data from request
        data = request.json
        
        # Extract parameters from the JSON request
        prompt = data['prompt']
        spatial_img_data = data['spatial_img']  # Image as base64 string
        height = int(data['height'])
        width = int(data['width'])
        seed = int(data['seed'])
        control_type = data['control_type']
        
        # Decode the base64 string to image
        spatial_img_bytes = base64.b64decode(spatial_img_data)
        spatial_img = Image.open(io.BytesIO(spatial_img_bytes))

        # Generate the image
        generated_image_base64 = single_condition_generate_image(prompt, spatial_img, height, width, seed, control_type)

        # Return the generated image in base64 format as JSON response
        return jsonify({'status': 'success', 'generated_image': generated_image_base64})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
