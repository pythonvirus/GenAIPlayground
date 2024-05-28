import boto3
import json

#system text prompt
prompt_data="""
Act as shakespeare and write a poem on Generative AI 
"""

bedrock=boto3.client(service_name="bedrock-runtime", region_name='us-east-1')

payload={ 
    "prompt":"Human:"+prompt_data+"Assistant:",
   # "anthropic_version":"bedrock-2023-05-31",
   # "max_tokens":2048,
    #"max_gen_len":512,
    "temperature":0.8,
    "max_tokens_to_sample":300,
    "top_k":250,
    "top_p":0.9,
    "stop_sequences": ["Human:"]

}

body=json.dumps(payload)
model_id="anthropic.claude-v2"

response=bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

response_body=json.loads(response.get("body").read())
response_text=response_body.get("completion")
print(response_text)