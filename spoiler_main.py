import openai
import time
import os
from dotenv import load_dotenv
load_dotenv()
# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def is_spoiler(review):
    prompt = f"Classify the following movie review as 'Spoiler' or 'Non-Spoiler':\n\n{review}\n\nAnswer:"

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Use "gpt-4-turbo" or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a movie review classifier that detects spoilers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0  # Low temperature for consistent responses
        )

        result = response.choices[0].message.content.strip()
        return result
    except Exception as e:
        #print(f"Error: {e}")
        return "Error"