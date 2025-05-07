from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
import json
import os
from dotenv import load_dotenv
from flask_cors import CORS
import google.generativeai as genai
import base64
from components.scoringModel import predict_score, train_from_user_data, train_dummy_model


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("Initializing scoring model...")
try:
    if os.path.exists("components/score_model.pkl"):
        os.remove("components/score_model.pkl")
    train_dummy_model()
    print("Scoring model created successfully")
except Exception as e:
    print(f"Error creating scoring model: {str(e)}")

app = Flask(__name__)
CORS(app) 
def fetch_website_content(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        raise Exception(f"Error fetching website: {str(e)}")

def determine_website_category(content):
    """Use Gemini API to determine website category."""
    try:
        limited_content = content[:2000]
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        You are an expert web analyst. Identify the most likely category of this website.
        
        Based on this website content, determine the category (e.g. e-commerce, blog, SaaS, portfolio, etc.):
        
        {limited_content}
        
        Return ONLY the category name, nothing else.
        """
        
        response = model.generate_content(prompt)
        
        category = response.text.strip()
        return category
    except Exception as e:
        raise Exception(f"Error determining website category: {str(e)}")

def extract_website_components(content, category):
    """Use Gemini API to extract website components and evaluate them."""
    try:
        limited_content = content[:3000]
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""
        You are an expert web analyst specializing in UX and conversion optimization.
        
        Analyze this {category} website content and extract the following components:
        
        1. CTA (Call to Action): Identify all CTAs and evaluate their effectiveness.
        2. Visual Hierarchy: Analyze how content is visually prioritized and structured.
        3. Copy Effectiveness: Evaluate the quality, clarity and persuasiveness of the text.
        4. Trust Signals: Identify elements that build trust (testimonials, certifications, etc).
        
        For each category, provide detailed observations. If any component is missing, note this as well.
        
        Format your response as JSON with the following structure:
        {{
            "cta": {{ "observations": [list of findings as simple strings] }},
            "visual_hierarchy": {{ "observations": [list of findings as simple strings] }},
            "copy_effectiveness": {{ "observations": [list of findings as simple strings] }},
            "trust_signals": {{ "observations": [list of findings as simple strings] }}
        }}
        
        Website content:
        {limited_content}
        
        Respond with ONLY the properly formatted JSON, nothing else. Each observation must be a simple string, not an object.
        """
        
        response = model.generate_content(prompt)
        
        try:
            analysis = json.loads(response.text)
        except json.JSONDecodeError:
            text = response.text
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                analysis = json.loads(json_str)
            else:
                analysis = {
                    "cta": {"observations": ["Unable to analyze CTAs"]},
                    "visual_hierarchy": {"observations": ["Unable to analyze visual hierarchy"]},
                    "copy_effectiveness": {"observations": ["Unable to analyze copy"]},
                    "trust_signals": {"observations": ["Unable to analyze trust signals"]}
                }
                
        return analysis
    except Exception as e:
        raise Exception(f"Error extracting website components: {str(e)}")

def generate_suggestions(analysis, category):
    """Generate prioritized improvement suggestions based on analysis."""
    try:
        analysis_json = json.dumps(analysis)
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""
        You are an expert conversion rate optimization consultant known for providing actionable suggestions.
        
        Based on this analysis of a {category} website:
        
        {analysis_json}
        
        Generate specific, actionable improvement suggestions for each component (CTA, Visual Hierarchy, Copy Effectiveness, Trust Signals).
        
        For each component:
        1. Provide at least 3 specific suggestions
        2. Rank each suggestion by impact potential (high, medium, low)
        3. Mark the 2 highest impact suggestions for each component
        
        Format your response as JSON with the following structure:
        {{
            "cta": {{
                "high_priority": [2 highest impact suggestions as simple strings],
                "additional": [remaining suggestions as simple strings]
            }},
            "visual_hierarchy": {{
                "high_priority": [2 highest impact suggestions as simple strings],
                "additional": [remaining suggestions as simple strings]
            }},
            "copy_effectiveness": {{
                "high_priority": [2 highest impact suggestions as simple strings],
                "additional": [remaining suggestions as simple strings]
            }},
            "trust_signals": {{
                "high_priority": [2 highest impact suggestions as simple strings],
                "additional": [remaining suggestions as simple strings]
            }}
        }}
        
        IMPORTANT: Each suggestion MUST be a simple string, not an object. Do not include impact ratings inside the arrays.
        
        Respond with ONLY the properly formatted JSON, nothing else.
        """
        
        response = model.generate_content(prompt)
        
        try:
            suggestions = json.loads(response.text)
        except json.JSONDecodeError:
            text = response.text
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                suggestions = json.loads(json_str)
            else:
                suggestions = {
                    "cta": {
                        "high_priority": ["Improve CTA visibility", "Make CTA messaging more compelling"],
                        "additional": ["Test different CTA colors"]
                    },
                    "visual_hierarchy": {
                        "high_priority": ["Improve content organization", "Enhance key element visibility"],
                        "additional": ["Add more whitespace between sections"]
                    },
                    "copy_effectiveness": {
                        "high_priority": ["Clarify value proposition", "Make headlines more compelling"],
                        "additional": ["Simplify complex sentences"]
                    },
                    "trust_signals": {
                        "high_priority": ["Add customer testimonials", "Display security badges"],
                        "additional": ["Include company credentials or awards"]
                    }
                }
        for section in suggestions:
            if 'high_priority' in suggestions[section]:
                suggestions[section]['high_priority'] = [
                    str(item) if not isinstance(item, str) else item 
                    for item in suggestions[section]['high_priority']
                ]
            
            if 'additional' in suggestions[section]:
                suggestions[section]['additional'] = [
                    str(item) if not isinstance(item, str) else item 
                    for item in suggestions[section]['additional']
                ]
                
        return suggestions
    except Exception as e:
        raise Exception(f"Error generating suggestions: {str(e)}")

@app.route('/components', methods=['POST'])
def analyze_website():
    """Main route to analyze a website from URL or HTML."""
    try:
        data = request.json

        if 'url' in data:
            content = fetch_website_content(data['url'])
            source = data['url']

            soup = BeautifulSoup(content, 'html.parser')
            for script in soup(["script", "style"]):
                script.extract()
            text_content = soup.get_text(separator=" ", strip=True)

            website_score = predict_score(html=content)
            
            return process_text_content(text_content, source, website_score)
        elif 'html' in data:
            content = data['html']
            source = "HTML input"

            soup = BeautifulSoup(content, 'html.parser')
            for script in soup(["script", "style"]):
                script.extract()
            text_content = soup.get_text(separator=" ", strip=True)

            website_score = predict_score(html=content)
            
            return process_text_content(text_content, source, website_score)
        elif 'image' in data:

            image_data = data['image']
            if ';base64,' in image_data:

                image_data = image_data.split(';base64,')[1]

            image_parts = [{"mime_type": "image/jpeg", "data": image_data}]
            
            website_score = None  
            
            return process_image_content(image_parts, "Image input", website_score)
        else:
            return jsonify({"error": "Either URL, HTML, or image is required"}), 400
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def process_text_content(text_content, source, website_score=None):
    """Process text content for analysis"""

    category = determine_website_category(text_content)
    
    components_analysis = extract_website_components(text_content, category)
    
    suggestions = generate_suggestions(components_analysis, category)
    
    result = {
        "source": source,
        "category": category,
        "analysis": components_analysis,
        "suggestions": suggestions,
        "website_score": website_score
    }
    
    return jsonify(result)

def process_image_content(image_parts, source, website_score=None):
    """Process image content for analysis"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        mime_type = image_parts[0]['mime_type']
        image_data = image_parts[0]['data']
        
        image_part = {
            "inline_data": {
                "mime_type": mime_type,
                "data": image_data
            }
        }

        category_prompt = "You are an expert web analyst. Identify the most likely category of this website screenshot (e.g. e-commerce, blog, SaaS, portfolio, etc.). Return ONLY the category name, nothing else."
        category_response = model.generate_content([category_prompt, image_part])
        category = category_response.text.strip()
        
        components_prompt = f"""
        You are an expert web analyst specializing in UX and conversion optimization.
        
        Analyze this {category} website screenshot and extract the following components:
        
        1. CTA (Call to Action): Identify all CTAs and evaluate their effectiveness.
        2. Visual Hierarchy: Analyze how content is visually prioritized and structured.
        3. Copy Effectiveness: Evaluate the quality, clarity and persuasiveness of the text.
        4. Trust Signals: Identify elements that build trust (testimonials, certifications, etc).
        
        For each category, provide detailed observations. If any component is missing, note this as well.
        
        Format your response as JSON with the following structure:
        {{
            "cta": {{ "observations": [list of findings as simple strings] }},
            "visual_hierarchy": {{ "observations": [list of findings as simple strings] }},
            "copy_effectiveness": {{ "observations": [list of findings as simple strings] }},
            "trust_signals": {{ "observations": [list of findings as simple strings] }}
        }}
        
        Respond with ONLY the properly formatted JSON, nothing else. Each observation must be a simple string, not an object.
        """
        
        components_response = model.generate_content([components_prompt, image_part])
        
        try:
            analysis = json.loads(components_response.text)
        except json.JSONDecodeError:
            text = components_response.text
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                analysis = json.loads(json_str)
            else:
               
                analysis = {
                    "cta": {"observations": ["Unable to analyze CTAs from image"]},
                    "visual_hierarchy": {"observations": ["Unable to analyze visual hierarchy from image"]},
                    "copy_effectiveness": {"observations": ["Unable to analyze copy from image"]},
                    "trust_signals": {"observations": ["Unable to analyze trust signals from image"]}
                }
        
        suggestions = generate_suggestions(analysis, category)
        
        total_observations = sum(len(analysis[key]['observations']) for key in analysis)
        positive_observations = 0
        negative_observations = 0
        for section in analysis:
            for observation in analysis[section]['observations']:
                lower_obs = observation.lower()
                if "unable to analyze" in lower_obs:
                    continue
                if any(term in lower_obs for term in ["missing", "lack", "no ", "poor", "weak", "confusing", "unclear", 
                                                     "ineffective", "absent", "could be", "should be", "not"]):
                    negative_observations += 1
                elif any(term in lower_obs for term in ["clear", "effective", "good", "strong", "well", "present", 
                                                      "prominent", "visible", "professional"]):
                    positive_observations += 1
        informative_observations = positive_observations + negative_observations
        if informative_observations > 0:
            image_based_score = 50 + (30 * (positive_observations / informative_observations - 0.5))
            image_based_score = min(100, max(0, image_based_score))
        else:
            image_based_score = 50
        website_score = image_based_score

        result = {
            "source": source,
            "category": category,
            "analysis": analysis,
            "suggestions": suggestions,
            "website_score": website_score
        }
        
        return jsonify(result)
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

@app.route('/train-model', methods=['POST'])
def train_scoring_model():
    """Endpoint to train the scoring model with user data and feedback"""
    try:
        data = request.json
        
        if not data or 'html' not in data or 'user_score' not in data:
            return jsonify({"error": "Missing required fields: html and user_score"}), 400
        
        html = data['html']
        user_score = float(data['user_score'])

        user_feedback = data.get('user_feedback', {})

        result = train_from_user_data(html, user_score, user_feedback)
        
        return jsonify({
            "success": True,
            "message": "Model trained successfully",
            "old_score": result["old_score"],
            "new_score": result["new_score"],
            "model_updated": result["model_updated"]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5050))
    app.run(host='0.0.0.0', port=port, debug=True)