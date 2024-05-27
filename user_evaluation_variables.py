import yaml
from yaml import safe_load
import streamlit as st

USERNAME = "ANONYMOUS"
EVAL_ID = None
MODEL = None
MODEL_TYPE = None
NO_SAMPLES = None
RESOLUTION = None
INFERENCE_STEPS = None
GEN_OBJECTS = None
GEN_ACTIONS = None
GEN_OCCUPATIONS = None
TASK_TARGET = None
DIST_BIAS = None
HALLUCINATION = None
MISS_RATE = None
DATE = None
TIME = None
RUN_TIME = None

EVAL_METRICS = None
OBJECT_IMAGES = []
OCCUPATION_IMAGES = []
TASK_IMAGES = []
OBJECT_CAPTIONS = None
OCCUPATION_CAPTIONS = None
TASK_CAPTIONS = None
TASK_COCOIDs = None

OBJECT_IMAGES_IN_UI = False
OCCUPATION_IMAGES_IN_UI = False
TASK_IMAGES_IN_UI = False
CURRENT_EVAL_TYPE = None
def update_evaluation_table(evalType, debugging):
    global USERNAME
    global EVAL_ID
    global MODEL
    global MODEL_TYPE
    global NO_SAMPLES
    global RESOLUTION
    global INFERENCE_STEPS
    global GEN_OBJECTS
    global GEN_ACTIONS
    global GEN_OCCUPATIONS
    global TASK_TARGET
    global DIST_BIAS
    global HALLUCINATION
    global MISS_RATE
    global DATE
    global TIME
    global RUN_TIME
    global CURRENT_EVAL_TYPE

    if debugging:
        st.write("Username: ", USERNAME)
        st.write("EVAL_ID: ", EVAL_ID)
        st.write("MODEL: ", MODEL)
        st.write("MODEL_TYPE: ", MODEL_TYPE)
        st.write("NO_SAMPLES: ", NO_SAMPLES)
        st.write("RESOLUTION: ", RESOLUTION)
        st.write("INFERENCE_STEPS: ", INFERENCE_STEPS)
        st.write("GEN_OBJECTS: ", GEN_OBJECTS)
        st.write("GEN_ACTIONS: ", GEN_ACTIONS)
        st.write("GEN_OCCUPATIONS: ", GEN_OCCUPATIONS)
        st.write("TASK_TARGET: ", TASK_TARGET)
        st.write("DIST_BIAS: ", DIST_BIAS)
        st.write("HALLUCINATION: ", HALLUCINATION)
        st.write("MISS_RATE: ", MISS_RATE)
        st.write("DATE: ", DATE)
        st.write("TIME: ", TIME)
        st.write("RUN_TIME: ", RUN_TIME)

    newEvaluationData = None
    if evalType == 'general':
        evalDataPath = './data/general_eval_database.yaml'
        newEvaluationData = {
            "Model": MODEL,
            "Model Type": MODEL_TYPE,
            "No. Samples": NO_SAMPLES,
            "Resolution": RESOLUTION,
            "Inference Steps": INFERENCE_STEPS,
            "Objects": GEN_OBJECTS,
            "Actions": GEN_ACTIONS,
            "Occupations": GEN_OCCUPATIONS,
            "Dist. Bias": DIST_BIAS,
            "Hallucination": HALLUCINATION,
            "Gen. Miss Rate": MISS_RATE,
            "Date": DATE,
            "Time": TIME,
            "Run Time": RUN_TIME
        }
    else:
        evalDataPath = './data/task_oriented_eval_database.yaml'
        newEvaluationData = {
            "Model": MODEL,
            "Model Type": MODEL_TYPE,
            "No. Samples": NO_SAMPLES,
            "Resolution": RESOLUTION,
            "Inference Steps": INFERENCE_STEPS,
            "Target": TASK_TARGET,
            "Dist. Bias": DIST_BIAS,
            "Hallucination": HALLUCINATION,
            "Gen. Miss Rate": MISS_RATE,
            "Date": DATE,
            "Time": TIME,
            "Run Time": RUN_TIME
        }
    with open(evalDataPath, 'r') as f:
        yamlData = safe_load(f)


    if TASK_TARGET is None:
        st.success('Congrats on your General Bias evaluation!', icon='\U0001F388')
    else:
        st.success('Congrats on your Task-Oriented Bias evaluation!', icon='\U0001F388')
    yamlData['evaluations']['username'][USERNAME]= {}

    yamlData['evaluations']['username'][USERNAME][EVAL_ID] = newEvaluationData

    if debugging:
        st.write("NEW DATABASE ", yamlData['evaluations']['username'][USERNAME])
    with open(evalDataPath, 'w') as yaml_file:
        yaml_file.write(yaml.dump(yamlData, default_flow_style=False))

def reset_variables(evalType):
    global USERNAME
    global EVAL_ID
    global MODEL
    global MODEL_TYPE
    global NO_SAMPLES
    global RESOLUTION
    global INFERENCE_STEPS
    global GEN_OBJECTS
    global GEN_ACTIONS
    global GEN_OCCUPATIONS
    global TASK_TARGET
    global DIST_BIAS
    global HALLUCINATION
    global MISS_RATE
    global DATE
    global TIME
    global RUN_TIME
    global EVAL_METRICS
    global OBJECT_IMAGES
    global OCCUPATION_IMAGES
    global TASK_IMAGES
    global OBJECT_CAPTIONS
    global OCCUPATION_CAPTIONS
    global TASK_CAPTIONS
    global TASK_COCOIDs
    global OBJECT_IMAGES_IN_UI
    global OCCUPATION_IMAGES_IN_UI
    global TASK_IMAGES_IN_UI
    global CURRENT_EVAL_TYPE
    EVAL_ID = None
    # MODEL = None
    # MODEL_TYPE = None
    NO_SAMPLES = None
    RESOLUTION = None
    INFERENCE_STEPS = None
    GEN_OBJECTS = None
    GEN_ACTIONS = None
    GEN_OCCUPATIONS = None

    DIST_BIAS = None
    HALLUCINATION = None
    MISS_RATE = None
    DATE = None
    TIME = None
    RUN_TIME = None

    EVAL_METRICS = None
    CURRENT_EVAL_TYPE = None

    if evalType == 'general':
        OBJECT_IMAGES = []
        OCCUPATION_IMAGES = []
        OBJECT_CAPTIONS = None
        OCCUPATION_CAPTIONS = None
        OBJECT_IMAGES_IN_UI = False
        OCCUPATION_IMAGES_IN_UI = False
    else:
        TASK_IMAGES = []
        TASK_CAPTIONS = None
        TASK_COCOIDs = None
        TASK_IMAGES_IN_UI = False
        TASK_TARGET = None

