import os
from openai import AzureOpenAI

import json

from dotenv import load_dotenv

load_dotenv()

print(os.getenv("AZURE_OPENAI_API_KEY"))

print(os.getenv("AZURE_OPENAI_ENDPOINT"))

print(os.getenv("deployment_name"))
    
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-02-01",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
deployment_name=os.getenv("deployment_name") #This will correspond to the custom name you chose for your deployment when you deployed a model. Use a gpt-35-turbo-instruct deployment. 

#print(api_key,azure_endpoint)    
# Send a completion call to generate an answer

print('Sending a test completion job')
start_phrase = 'what is Multimodel RAG'
response = client.completions.create(model=deployment_name, prompt=start_phrase, max_tokens=100)
#print(response)
print(response.choices[0].text)




