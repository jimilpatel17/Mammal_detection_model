import os
from google import genai
from google.genai import types
from PIL import Image
import io

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def predict_mammal_with_gemini(image_path):
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        prompt = """You are a wildlife expert focused on mammals of the Indian subcontinent.
Identify the species shown in the image, and provide ONLY the scientific name.

**Important**: Only identify mammals native to or commonly found in the Indian subcontinent (India, Pakistan, Bangladesh, Sri Lanka, Nepal, Bhutan).
If the image shows an animal that is not a mammal or not from this region, respond with "Not identifiable as a native mammal of the Indian subcontinent."

Provide only the scientific name in the format: Genus_species (e.g., "Panthera_tigris")"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                ),
                prompt
            ]
        )
        
        if response.text:
            species_name = response.text.strip().replace(' ', '_')
            return {
                'species': species_name.replace('_', ' '),
                'source': 'Gemini AI'
            }
        else:
            return {
                'species': 'Unable to identify',
                'source': 'Gemini AI'
            }
    
    except Exception as e:
        return {
            'species': f'Error: {str(e)}',
            'source': 'Gemini AI'
        }
