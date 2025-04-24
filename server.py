from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64
import os
import time

api_key = os.getenv('GEMINI_API_KEY')
output_path = os.getenv('OUTPUT_IMAGE_PATH')
client = genai.Client(api_key=api_key)

# server.py
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Demo")

@mcp.tool()
def generate_image(prompt: str) -> str:
    """Generate an image from a prompt"""
    contents = [prompt]
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp-image-generation",
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE']
        )
    )

    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image = Image.open(BytesIO((part.inline_data.data)))
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            image_path = os.path.join(output_path, f"gemini_image_{int(time.time())}.png")
            image.save(image_path)
            image = Image.open(image_path)  # Reopen the saved image
            image.show()  # Show the image
            print(f"Image saved to: {image_path}")
            return image_path
    
    return "No image generated"


    


