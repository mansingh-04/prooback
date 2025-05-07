import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_ai_feedback(html=None, image_data=None):
    """
    Get AI feedback on website UX/UI based on HTML or image input
    Args:
        html: HTML content as string
        image_data: Base64 encoded image data
    """
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    if html:
        prompt = f"""You're a UX expert. Analyze the following landing page HTML and suggest 3 improvements focused on:
- CTA clarity
- Visual hierarchy
- Copy effectiveness
- Trust signals

HTML:
{html[:3000]}  # limit to avoid token overload
"""
        try:
            response = model.generate_content(prompt)
            return response.text.split("\n")
        except Exception as e:
            return [f"AI feedback failed: {str(e)}"]
    
    elif image_data:
        prompt = """You're a UX expert. Analyze the following landing page screenshot and suggest 3 improvements focused on:
- CTA clarity
- Visual hierarchy
- Copy effectiveness
- Trust signals
"""
        if ';base64,' in image_data:
            image_data = image_data.split(';base64,')[1]
        image_part = {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": image_data
            }
        }
        
        try:
            response = model.generate_content([prompt, image_part])
            return response.text.split("\n")
        except Exception as e:
            return [f"AI feedback from image failed: {str(e)}"]
            
    else:
        return ["Error: No input provided. Please provide either HTML or image data."]
