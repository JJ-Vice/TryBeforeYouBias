import torch
import requests
import urllib.request
import streamlit as st

validT2IModelTypes = ["KandinskyPipeline", "StableDiffusionPipeline", "DiffusionPipeline", "StableDiffusionXLPipeline",
                      "LatentConsistencyModelPipeline"]
def check_if_model_exists(repoName):
    modelLoaded = None
    huggingFaceURL = "https://huggingface.co/" + repoName + "/raw/main/model_index.json"
    response = requests.get(huggingFaceURL).status_code
    if response != 200:
        return None
    else:
        return huggingFaceURL
      
def get_model_info(modelURL):
    modelType = None
    try:
        with urllib.request.urlopen(modelURL) as f:
            modelType = str(f.read()).split(',\\n')[0].split(':')[1].replace('"', '').strip()
    except urllib.error.URLError as e:
        st.write(e.reason)
    return modelType

# Definitely need to work on these functions to consider adaptors
# currently only works if there is a model index json file

def import_model(modelID, modelType):
    T2IModel = None    
    if modelType in validT2IModelTypes:
        if modelType == 'StableDiffusionXLPipeline':
            from diffusers import StableDiffusionXLPipeline
            T2IModel = StableDiffusionXLPipeline.from_pretrained(modelID, torch_dtype=torch.float16)
        elif modelType == 'LatentConsistencyModelPipeline':
            from diffusers import DiffusionPipeline
            T2IModel = DiffusionPipeline.from_pretrained(modelID, torch_dtype=torch.float16)
        else:
            from diffusers import AutoPipelineForText2Image
            T2IModel = AutoPipelineForText2Image.from_pretrained(modelID, torch_dtype=torch.float16)
        T2IModel.to("cuda")
    return T2IModel
