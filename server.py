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
client = genai.Client(api_key=api_key)

# server.py
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Gemini Image Generator")

@mcp.tool()
def generate_image(prompt: str, save_path: str, filename: Optional[str] = None) -> str:
    """Generate an image from a text prompt using Google's Gemini model.

    Args:
        prompt (str): The text description of the image to generate.
        save_path (str): Directory path where the image should be saved.
        filename (Optional[str]): Optional desired filename for the image (without extension). Defaults to a timestamp-based name.

    Returns:
        str: Path to the generated image file, or error message if generation failed.

    Raises:
        Exception: If image generation fails.
    """
    try:
        # Ensure output directory exists
        os.makedirs(save_path, exist_ok=True)

        # Generate image - using generate_image (singular) as per test.py
        response = client.models.generate_image(
            model="imagen-3.0-generate-002",
            prompt=prompt,
            config=types.GenerateImageConfig( # Using GenerateImageConfig (singular) as per test.py
                number_of_images=1, # Still generating 1 for this tool
            )
        )

        # Process response - checking generated_images as per test.py
        if hasattr(response, 'generated_images') and response.generated_images:
            generated_image = response.generated_images[0] # Get the first image
            
            # Access image bytes as per test.py
            if hasattr(generated_image, 'image') and hasattr(generated_image.image, 'image_bytes'):
                 image_bytes = generated_image.image.image_bytes
            else:
                 # Fallback or error handling if structure differs unexpectedly
                 return "Could not find image data in the expected format"

            # Save image
            image = Image.open(BytesIO(image_bytes))
            base_filename = filename if filename else f"gemini_image_{int(time.time())}"
            image_path = os.path.join(save_path, f"{base_filename}.png")
            image.save(image_path)
            print(f"Image saved to: {image_path}")
            return image_path
        else:
            # Check for other possible response structures or return error
            # (Simplified error handling for now, can add fallbacks later if needed)
            print("No image data found in response.")
            return "No image was generated in the response"

    except Exception as e:
        error_msg = f"Failed to generate image: {str(e)}"
        print(error_msg)
        return error_msg
