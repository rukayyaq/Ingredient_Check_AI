# ocr_processor.py
import requests
import base64
import os
import json
from typing import Tuple, Optional
import config

class OCRProcessor:
    def __init__(self):
        self.api_key = config.GEMINI_API_KEY
        # Using the standard v1beta model endpoint
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"

    def extract_text_from_image(self, image_path) -> Tuple[Optional[str], str]:
        """
        Takes a direct FILE PATH string and sends it DIRECTLY to Gemini's API 
        using a standard HTTP request.
        """
        if not image_path or not os.path.exists(image_path):
            return None, "Unknown Product"

        try:
            # 1. Read the image and encode it to Base64
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            # 2. Prepare the JSON payload for Google's API
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": "Read this image and extract the exact text. Do not add any conversational commentary. Just return the raw text."
                            },
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": encoded_image
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.0,
                    "topK": 1,
                    "topP": 1,
                    "maxOutputTokens": 2048
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    }
                ]
            }

            # 3. Send the request
            headers = {'Content-Type': 'application/json'}
            response = requests.post(self.url, headers=headers, json=payload)
            
            # 4. Parse the response
            if response.status_code == 200:
                data = response.json()
                try:
                    text_result = data['candidates'][0]['content']['parts'][0]['text']
                    print(f"\n--- DIRECT API SUCCESS ---\n{text_result}\n--------------------------\n")
                    return text_result.strip(), "Analyzed Product"
                except (KeyError, IndexError):
                    print(f"API returned JSON but no text: {data}")
                    return "", "Unknown Product"
            else:
                print(f"API ERROR {response.status_code}: {response.text}")
                return "", "Unknown Product"
                
        except Exception as e:
            print(f"CRITICAL ERROR IN OCR: {e}")
            return "", "Unknown Product"