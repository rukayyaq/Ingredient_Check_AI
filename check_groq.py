import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

print("=" * 60)
print("GROQ API / MODEL CHECK")
print("=" * 60)

if not api_key:
    print("ERROR: GROQ_API_KEY was not found.")
    exit()

print("API key found: YES")
print("API key starts with:", api_key[:8] + "...")
print()

try:
    client = Groq(api_key=api_key)

    models = client.models.list()

    print("AVAILABLE MODELS:")
    print("=" * 60)

    for model in models.data:
        print(model.id)

    print("=" * 60)
    print("Total models:", len(models.data))

except Exception as e:
    print("ERROR:")
    print(e)