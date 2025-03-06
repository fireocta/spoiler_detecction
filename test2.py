import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

models = openai.models.list()
for model in models:
    print(model.id)
