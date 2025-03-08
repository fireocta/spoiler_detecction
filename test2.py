""" import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

models = openai.models.list()
for model in models:
    print(model.id)
 """

from spoiler_main import is_spoiler

a=is_spoiler("What a great movie! I loved the plot twist at the end.")
if(a=="Spoiler"):
    print("It's a spoiler")
elif (a=="Non-Spoiler"):
    print("It's not a spoiler")
else:
    print("Error")
