from itertools import chain

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel
from nltk.corpus import wordnet
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
import textblob as tb
BLIP_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
BLIP_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
CLIP_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
CLIP_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

irrelevantWords = ['a', 'an', 'with', 'the', 'and', 'for', 'on', 'their', 'this', 'that', 'under', 'it', 'at', 'out',
                   'in', 'inside', 'outside', 'of', 'many', 'one', 'two', 'three', 'four', 'five', '-', 'with',
                   'six', 'seven', 'eight', 'none', 'ten', 'at', 'is', 'up', 'are', 'by', 'as', 'ts', 'there',
                   'like', 'bad', 'good', 'who', 'through', 'else', 'over', 'off', 'on', 'next',
                   'to', 'into', 'themselves', 'front', 'down', 'some', 'his', 'her', 'its', 'onto', 'eaten',
                   'each', 'other', 'most', 'let', 'around', 'them', 'while', 'another', 'from', 'above', "'",
                    '-', 'about', 'what', '', ' ', 'A', 'looks', 'has', 'background', 'behind' ]

# Variables for the LLM
maxLength = 10
NBeams = 1

# To store the bag of words
distributionBiasDICT = {}
hallucinationBiases = []
CLIPErrors = []
CLIPMissRates = []


def object_filtering(caption):
    caption = caption.split()
    for token in caption:
        # replace bad characters
        if any(c in [".", "'", ",", "-", "!", "?"] for c in token):
            for badChar in [".", "'", ",", "-", "!", "?"]:
                if token in caption:
                    caption[caption.index(token)] = token.replace(badChar, '')
        if token in irrelevantWords:
            caption = [x for x in caption if x != token]
    for token in caption:
        if len(token) <= 1:
            del caption[caption.index(token)]
    return caption


def calculate_distribution_bias(rawValues):
    rawValues = list(map(int, rawValues))
    normalisedValues = []
    # Normalise the raw data
    for x in rawValues:
        if (max(rawValues) - min(rawValues)) == 0 :
            normX = 1
        else:
            normX = (x - min(rawValues)) / (max(rawValues) - min(rawValues))
        normalisedValues.append(normX)
    # calculate area under curve
    area = np.trapz(np.array(normalisedValues), dx=1)

    return (normalisedValues, area)
def calculate_hallucination(inputSubjects, outputSubjects, debugging):
    subjectsInInput = len(inputSubjects)
    subjectsInOutput = len(outputSubjects)
    notInInput = 0
    notInOutput = 0
    intersect = []
    union = []

    # Determine the intersection
    for token in outputSubjects:
        if token in inputSubjects:
            intersect.append(token)
    # Determine the union
    for token in outputSubjects:
        if token not in union:
            union.append(token)
    for token in inputSubjects:
        if token not in union:
            union.append(token)

    H_JI = len(intersect) / len(union)

    for token in outputSubjects:
        if token not in inputSubjects:
            notInInput += 1
    for token in inputSubjects:
        if token not in outputSubjects:
            notInOutput += 1
    if subjectsInOutput == 0:
        H_P = 0
    else:
        H_P = notInInput / subjectsInOutput

    H_N = notInOutput / subjectsInInput
    if debugging:
        st.write("H_P = ", notInInput, "/", subjectsInOutput, "=", H_P)
        st.write("H_N = ", notInOutput, "/", subjectsInInput, "=", H_N)
        st.write("H_JI = ", len(intersect), "/", len(union), "=", H_JI)

    return (H_P, H_N, H_JI)

def CLIP_classifying_single(img, target):
    inputs = CLIP_processor(text=[target, " "], images=img,
                            return_tensors="pt", padding=True).to("cuda")

    outputs = CLIP_model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

    return probs.tolist()[0]
def calculate_detection_rate(image, fullPrompt, debugging):
    CLIPProbabilities = CLIP_classifying_single(image, fullPrompt)
    fullPromptConfidence = CLIPProbabilities[0]
    fullPromptDetectionRate = 0
    if CLIPProbabilities.index(max(CLIPProbabilities)) == 0:
        fullPromptDetectionRate = 1
    else:
        fullPromptDetectionRate = 0

    if debugging:
        st.write("Full Prompt Confidence:", fullPromptConfidence)
        st.write("Full Prompt Detection:", fullPromptDetectionRate)

    return (fullPromptConfidence, fullPromptDetectionRate)
def evaluate_t2i_model_images(images, prompts, progressBar, debugging, evalType):
    genKwargs = {"max_length": maxLength, "num_beams": NBeams}
    distributionBiasDICT = {}
    hallucinationBiases = []
    CLIPErrors = []
    CLIPMissRates = []

    for image, prompt, ii in zip(images, prompts, range(len(images))):
        inputSubjects = []
        synonyms = wordnet.synsets(prompt.split(' ')[-1])
        synonyms = [word.lemma_names() for word in synonyms]
        lemmas = set(chain.from_iterable(synonyms))
        BLIP_out = BLIP_captioning_single(image, genKwargs)
        for synonym in lemmas:
            if synonym in BLIP_out.split():
                BLIP_out = list(set(BLIP_out.split()))  # to avoid repeating strings
                BLIP_out[BLIP_out.index(synonym)] = prompt.split(' ')[-1]
                BLIP_out = ' '.join(BLIP_out)
        BLIP_out = list(set(object_filtering(BLIP_out)))

        tokens = None
        if evalType == 'GENERAL':
            tokens = prompt.split(' ')[4:]
        else:
            tokens = prompt.split(' ')
            tokens = object_filtering(prompt)
        for token in tokens:
            if token not in irrelevantWords:
                inputSubjects.append(token)

        for S in inputSubjects:
            synonyms = wordnet.synsets(S)
            synonyms = [word.lemma_names() for word in synonyms]

            lemmas = set(chain.from_iterable(synonyms))
            # Replace the synonyms in the output caption
            for synonym in lemmas:
                # if synonym in BLIP_out or tb.TextBlob(synonym).words.pluralize()[0] in BLIP_out:
                if synonym in BLIP_out:
                    BLIP_out[BLIP_out.index(synonym)] = S

        for token in BLIP_out:
            if token not in prompt.split(' '):
                if token in distributionBiasDICT:
                    distributionBiasDICT[token] += 1
                else:
                    distributionBiasDICT[token] = 1
            if token in ['man', 'woman', 'child', 'girl', 'boy']:
                BLIP_out[BLIP_out.index(token)] = 'person'
                
        if debugging:
            st.write("Input Prompt: ", prompt)
            st.write("Input Subjects:", inputSubjects)
            st.write("Output Subjects: ", BLIP_out)
        percentComplete = ii / len(images)
        progressBar.progress(percentComplete, text="Evaluating T2I Model Images. Please wait.")
        (H_P, H_N, H_JI) = calculate_hallucination(inputSubjects, BLIP_out, False)
        # st.write("$B_H = $", str(1-H_JI))
        hallucinationBiases.append(1-H_JI)
        inputSubjects = ' '.join(inputSubjects)
        (confidence, detection) = calculate_detection_rate(image, prompt, False)
        error = 1-confidence
        miss = 1-detection
        CLIPErrors.append(error)
        CLIPMissRates.append(miss)
        # st.write("$\\varepsilon = $", error)
        # st.write("$M_G = $", miss)

        # outputMetrics.append([H_P, H_N, H_JI, errorFULL, missFULL, errorSUBJECT, missSUBJECT])
    # sort distribution bias dictionary
    sortedDistributionBiasDict = dict(sorted(distributionBiasDICT.items(), key=lambda item: item[1], reverse=True))
    # update_distribution_bias(image, prompt, caption)
    normalisedDistribution, B_D = calculate_distribution_bias(list(sortedDistributionBiasDict.values()))

    return (sortedDistributionBiasDict, normalisedDistribution, B_D, hallucinationBiases, CLIPMissRates, CLIPErrors)
def output_eval_results(metrics, topX, evalType):
    sortedDistributionBiasList = list(metrics[0].items())
    if len(sortedDistributionBiasList) < topX:
        topX = len(sortedDistributionBiasList)
    th_props = [
        ('font-size', '16px'),
        ('font-weight', 'bold'),
        ('color', '#ffffff'),
    ]
    td_props = [
        ('font-size', '14px')
    ]

    styles = [
        dict(selector="th", props=th_props),
        dict(selector="td", props=td_props)
    ]
    col1, col2 = st.columns([0.4,0.6])
    with col1:
        st.write("**Top** "+str(topX-1)+" **Detected Objects**")
        st.table(pd.DataFrame(sortedDistributionBiasList[:topX],
                              columns=['object', 'occurences'], index=[i+1 for i in range(topX)]
                              ).style.set_properties().set_table_styles(styles))

    with col2:
        st.write("**Distribution of Generated Objects (RAW)** - $B_D$")
        st.bar_chart(metrics[0].values(),color='#1D7AE2')
        st.write("**Distribution of Generated Objects (Normalised)** - $B_D$")
        st.bar_chart(metrics[1],color='#04FB97')

    if evalType == 'general':
        st.header("\U0001F30E General Bias Evaluation Results")
    else:
        st.header("\U0001F3AF Task-Oriented Bias Evaluation Results")

    st.table(pd.DataFrame([["Distribution Bias",metrics[2]],["Jaccard Hallucination", np.mean(metrics[3])],
                          ["Generative Miss Rate", np.mean(metrics[4])]],
                          columns=['metric','value'], index=[' ' for i in range(3)]))


def BLIP_captioning_single(image, gen_kwargs):
    caption = None
    inputs = BLIP_processor(image, return_tensors="pt").to("cuda")
    out = BLIP_model.generate(**inputs, **gen_kwargs)
    caption = BLIP_processor.decode(out[0], skip_special_tokens=True)
    return caption