from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64
import os
import time
from typing import Optional

# Get environment variables with error checking
def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value

# Initialize Gemini client
api_key = get_required_env('GEMINI_API_KEY')
output_path = get_required_env('OUTPUT_IMAGE_PATH')
client = genai.Client(api_key=api_key)

# server.py
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Gemini Image Generator")

@mcp.tool()
def generate_image(prompt: str) -> str:
    """Generate an image from a text prompt using Google's Gemini model.
    
    Args:
        prompt (str): The text description of the image to generate
        
    Returns:
        str: Path to the generated image file, or error message if generation failed
        
    Raises:
        Exception: If image generation fails
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Generate image
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )

        # Process response
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                print(f"Gemini says: {part.text}")
            elif part.inline_data is not None:
                # Save and display image
                image = Image.open(BytesIO((part.inline_data.data)))
                image_path = os.path.join(output_path, f"gemini_image_{int(time.time())}.png")
                image.save(image_path)
                image = Image.open(image_path)
                image.show()
                print(f"Image saved to: {image_path}")
                return image_path
        
        return "No image was generated in the response"
        
    except Exception as e:
        error_msg = f"Failed to generate image: {str(e)}"
        print(error_msg)
        return error_msg


    


