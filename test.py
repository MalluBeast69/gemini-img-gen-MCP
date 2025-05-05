from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import os
import json

client = genai.Client(api_key='AIzaSyC9guhe1VcFsmDLyAXpc9aXlP2UqoWQ0Wk')

# Define a function to generate images with optional save path
def generate_image_content_gen(prompt, save_path=None):
    print(f"Generating image for prompt: {prompt}")

    response = client.models.generate_image(
        model='imagen-3.0-generate-002',
        prompt=prompt,
        config=types.GenerateImageConfig(
            number_of_images=1,
        )
    )

    # Try different possible response structures
    if hasattr(response, 'images'):
        images = response.images
    elif hasattr(response, 'image'):
        images = [response.image]
    elif hasattr(response, 'generated_images'):
        images = response.generated_images
    else:
        return "Could not find images in response"

    if images and len(images) > 0:
        try:
            first_image = images[0]

            # Try different possible image data attributes
            if hasattr(first_image, 'data'):
                image_data = first_image.data
            elif hasattr(first_image, 'image_bytes'):
                image_data = first_image.image_bytes
            elif hasattr(first_image, 'image') and hasattr(first_image.image, 'image_bytes'):
                image_data = first_image.image.image_bytes
            else:
                return "Could not find image data"

            # Open the image
            image = Image.open(BytesIO(image_data))

            # Save the image if a path is provided
            if save_path:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                image.save(save_path)
                return save_path

            # Display the image
            image.show()
            return "Image generated but not saved"
        except Exception as e:
            return f"Error: {e}"
    else:
        return "No images were generated"

# Test the function
output_path = generate_image_content_gen("a beautiful mountain", "output/sunset.jpg")
print(output_path)