import cv2
from PIL import Image
from gtts import gTTS
import os
from dotenv import load_dotenv
import easyocr
import requests
from tkinter import filedialog
from tkinter import Tk

# Load environment variables
load_dotenv()

# Gemini API Key from .env file
GEMINI_API_KEY = os.getenv("AIzaSyCcM67sJkLs9JnIZtHfk4APzyTTO0hE-PM")

def upload_image():
    # Open file dialog to select an image
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select an Image", 
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    
    if not file_path:
        print("No image selected!")
        return None
    
    print(f"Image selected: {file_path}")
    return file_path

def extract_text(image_path):
    try:
        # Initialize the EasyOCR reader
        reader = easyocr.Reader(['en'])  # 'en' for English

        # Read text from the image
        result = reader.readtext(image_path)

        # Extract and concatenate the text from the result
        text = ""
        for detection in result:
            text += detection[1] + "\n"  # detection[1] contains the text
        
        return text.strip()  # Return the concatenated text, removing any extra spaces

    except Exception as e:
        print(f"Error reading image: {e}")
        return None

def query_gemini_api(question):
    url = "https://api.gemini.ai/query"  # Double-check the URL in the official documentation
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}"}
    payload = {"question": question}
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json().get("answer", "No answer found.")
        else:
            print("Error querying Gemini API:", response.status_code, response.text)
            return None
    except Exception as e:
        print(f"Error querying Gemini API: {e}")
        return None

def text_to_speech(text, output_file="output.mp3"):
    tts = gTTS(text=text, lang='en')
    tts.save(output_file)
    print(f"Audio saved as {output_file}")
    # Optionally, use a library like playsound or pygame to play the audio instead of relying on os.system
    os.system(f"start {output_file}" if os.name == "nt" else f"open {output_file}")

def main():
    # Step 1: Upload an image
    image_path = upload_image()
    if not image_path:
        return

    # Step 2: Extract text from the image
    question = extract_text(image_path)
    if not question:
        return

    # Step 3: Query the Gemini API
    answer = query_gemini_api(question)
    if not answer:
        return

    print("Answer:", answer)

    # Step 4: Convert the answer to speech
    text_to_speech(answer)

    print("Process completed.")

if __name__ == "__main__":
    main()
