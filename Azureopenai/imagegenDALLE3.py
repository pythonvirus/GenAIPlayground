import os
from openai import AzureOpenAI

import json

from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-02-01",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
deployment_name=os.getenv("deployment_name") 

result = client.images.generate(
    model="Dalle3", # the name of your DALL-E 3 deployment
    prompt="Give me closeup image of fish who is dancing in water.",
    n=1
)

image_url = json.loads(result.model_dump_json())['data'][0]['url']

print(image_url)