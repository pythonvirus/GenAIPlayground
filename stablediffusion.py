import boto3
import json
import base64
import os
from datetime import datetime

prompt_data="""

Do not use write title words in the image.

Please make you do not generate add any text while generating the image. Image should have objects but no written text.

Generate image using above guidelines for elearning course title "HR guidelines for employees"

"""

prompt_template=[{"text":prompt_data,"weight":1}]
bedrock=boto3.client(service_name="bedrock-runtime", region_name='us-east-1')

payload={
    "text_prompts":prompt_template,
    "cfg_scale":10,
    "seed":0,
    "steps":50,
    "width":1024,
    "height":1024,
    "style_preset" : "photographic"
}

body=json.dumps(payload)
model_id="stability.stable-diffusion-xl-v1"

response=bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

start_time = timer(None) # timing starts from this point for "start_time" variable
response_body=json.loads(response.get("body").read())
print("image generated successfully")
timer(start_time) 
#print(response_body)
artifact = response_body.get("artifacts")[0]
image_encoded = artifact.get("base64").encode("utf-8")
image_bytes = base64.b64decode(image_encoded)

# Save image to a file in the output directory.
output_dir = "output"
#os.makedirs(output_dir, exist_ok=True)
file_name = f"{output_dir}/generated-img.png"
with open(file_name, "wb") as f:
    f.write(image_bytes)

print("image saved successfully")


