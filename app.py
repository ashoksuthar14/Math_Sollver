import os
import base64
from PIL import Image
import io
import pytesseract
import google.generativeai as genai
from google.cloud import texttospeech
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import time

# Load environment variables from the .env file
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API"))

# Initialize Google Cloud TTS client
tts_client = texttospeech.TextToSpeechClient()

# Flask app setup
app = Flask(__name__, static_url_path='/static')

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/process-frame', methods=['POST'])
def process_frame():
    try:
        print("Processing new frame...")  # Debug log
        
        # Get the base64 image from the request
        data = request.get_json()
        base64_image = data['image'].split(',')[1]
        
        # Convert base64 to image
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))
        
        # Optimize image preprocessing for better OCR
        # Convert to grayscale
        image = image.convert('L')
        
        # Optimize size - make it smaller for faster processing
        target_width = 800  # Reduced from 1000
        ratio = target_width / image.width
        target_height = int(image.height * ratio)
        image = image.resize((target_width, target_height), Image.Resampling.BILINEAR)
        
        # Enhance contrast and sharpness
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)
        
        # Optimize Tesseract configuration for speed and accuracy
        custom_config = r'--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789+-*/()=. -c tessedit_pageseg_mode=1'
        
        try:
            # Increase timeout and add error handling for Tesseract
            extracted_text = pytesseract.image_to_string(
                image,
                config=custom_config,
                timeout=30  # Increased timeout to 30 seconds
            )
        except RuntimeError as e:
            if "timeout" in str(e):
                return jsonify({"error": "Image processing took too long. Please try a clearer image."}), 400
            raise
        
        # Clean up extracted text more efficiently
        extracted_text = ''.join(c for c in extracted_text if c in '0123456789+-*/()=. \n')
        extracted_text = extracted_text.strip()
        print(f"Extracted text: {extracted_text}")
        
        if not extracted_text or extracted_text.isspace():
            return jsonify({"solution": "No text detected in the image. Please ensure the math problem is clearly visible.", "hasAudio": False})

        # Optimize prompt for faster response
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""Solve this math problem step by step: {extracted_text}
        Format: 
        Problem: [problem]
        Step 1: [step]
        Step 2: [step]
        Final Answer: [answer]
        Keep explanations brief and clear."""
        
        response = model.generate_content(prompt)
        solution = response.text

        if "INVALID_EXPRESSION" in solution:
            return jsonify({"solution": "No valid math problem detected", "hasAudio": False})

        # Clean up the text for more natural speech
        speech_text = solution
        speech_text = speech_text.replace('*', 'multiplied by')
        speech_text = speech_text.replace('/', 'divided by')
        speech_text = speech_text.replace('-', 'minus')
        speech_text = speech_text.replace('+', 'plus')
        speech_text = speech_text.replace('=', 'equals')
        speech_text = speech_text.replace('Step', '\nStep')
        
        # Ensure static directory exists
        os.makedirs('static', exist_ok=True)
        
        # Generate unique filename for audio
        audio_filename = f"solution_{int(time.time())}.mp3"
        audio_path = os.path.join(os.getcwd(), 'static', audio_filename)
        
        print(f"Saving audio to: {audio_path}")  # Debug log
        
        # Convert solution to speech using Google Cloud TTS
        synthesis_input = texttospeech.SynthesisInput(
            ssml=f"""
            <speak>
                <prosody rate="0.9">
                    {speech_text}
                </prosody>
            </speak>
            """
        )
        
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Studio-O",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.95,
            effects_profile_id=['large-home-entertainment-class-device']
        )

        tts_response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        # Save the audio file
        with open(audio_path, "wb") as out:
            out.write(tts_response.audio_content)
        
        print("Audio file saved successfully")  # Debug log
        
        # Verify file exists
        if not os.path.exists(audio_path):
            raise Exception("Audio file was not saved correctly")
            
        return jsonify({
            "solution": solution,
            "hasAudio": True,
            "audioFile": audio_filename,
            "debug": {
                "audioPath": audio_path,
                "fileExists": os.path.exists(audio_path),
                "fileSize": os.path.getsize(audio_path)
            }
        })

    except Exception as e:
        print(f"Error: {str(e)}")  # Debug log
        import traceback
        traceback.print_exc()  # Print full error traceback
        return jsonify({"error": f"Error processing image: {str(e)}"}), 400

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)
