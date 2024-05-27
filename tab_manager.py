import streamlit as st
import model_inferencing as MINFER
import general_bias_measurement as GBM
import model_comparison as MCOMP
import user_evaluation_variables
import pandas as pd
import numpy as np
import json
import csv
from itertools import cycle
import random
import string
import time
import datetime
import zipfile
from io import BytesIO, StringIO
def completed_setup(tabs, modelID):
    with tabs[0]:
        st.write("\U0001F917 ", modelID, " has been loaded!")
        st.write("Ready for General Bias Evaluation")
        # general_bias_eval_setup(tabs[0])
    with tabs[1]:
        st.write("\U0001F917 ", modelID, " has been loaded!")
        st.write("Ready for Task-Oriented Bias Evaluation")
    with tabs[3]:
        if not all([user_evaluation_variables.OBJECT_IMAGES_IN_UI, user_evaluation_variables.OCCUPATION_IMAGES_IN_UI, user_evaluation_variables.TASK_IMAGES_IN_UI]):
            st.write("\U0001F917 ", modelID, " has been loaded!")
            st.write("Waiting for Images to be generated.")
        # if any([user_evaluation_variables.OBJECT_IMAGES_IN_UI, user_evaluation_variables.OCCUPATION_IMAGES_IN_UI,
        #             user_evaluation_variables.TASK_IMAGES_IN_UI]):
        update_images_tab(tabs[3])
    with tabs[0]:
        general_bias_eval_setup(tabs[0], modelID, tabs[3])
    with tabs[1]:
        task_oriented_bias_eval_setup(tabs[1],modelID, tabs[3])
def general_bias_eval_setup(tab, modelID, imagesTab):

    generalBiasSetupDF_EVAL = pd.DataFrame(
        {
            "GEN Eval. Variable": ["No. Images to Generate per prompt", "No. Inference Steps",
                                   "Image Height - must be a value that is 2 to the power of N",
                                   "Image Width - must be a value that is 2 to the power of N"],
            "GEN Values": ["2", "10", "512", "512"],
        }
    )
    generalBiasSetupDF_TYPE = pd.DataFrame(
        {
            "Image Types": ["Objects", "Person in Frame", "Occupations / Label"],
            "Check": [True, True, True],
        }
    )
    tableColumn1, tableColumn2 = st.columns(2)
    with tab:
        with tableColumn1:
            GENValTable = st.data_editor(
                generalBiasSetupDF_EVAL,
                column_config={
                    "GEN Eval. Variable": st.column_config.Column(
                        "Variable",
                        help="General Bias Evaluation variable to control extent of evaluations",
                        width=None,
                        required=None,
                        disabled=True,
                    ),
                    "GEN Values": st.column_config.Column(
                        "Values",
                        help="Input values in this column",
                        width=None,
                        required=True,
                        disabled=False,
                    ),
                },
                hide_index=True,
                num_rows="fixed",
            )
        with tableColumn2:
            GENCheckTable = st.data_editor(
                generalBiasSetupDF_TYPE,
                column_config={
                    "Check": st.column_config.CheckboxColumn(
                        "Select",
                        help="Select the types of images you want to generate",
                        default=False,
                    )
                },
                disabled=["Image Types"],
                hide_index=True,
                num_rows="fixed",
            )
        st.info(
            'Image sizes vary for each model but is generally one of [256, 512, 1024, 2048]. We found that for some models '
            'lower image resolutions resulted in noise outputs (you are more than welcome to experiment with this). '
            'Consult the model card if you are unsure what image resolution to use.', icon="ℹ️")
        if not all([GENValTable["GEN Values"][0].isnumeric(), GENValTable["GEN Values"][1].isnumeric(), GENValTable["GEN Values"][2].isnumeric()]):
            st.error('Looks like you have entered non-numeric values! '
                     'Please enter numeric values in the table above', icon="🚨")
        # elif not all([check_for_power_of_two(int(GENValTable["GEN Values"][2])), int(GENValTable["GEN Values"][2]) >= 8]):
        #     st.error('Please ensure that your image resolution is 1 number greater than or equal to 8', icon="🚨")
        else:
            if st.button('Evaluate!', key="EVAL_BUTTON_GEN"):
                initiate_general_bias_evaluation(tab, modelID, [GENValTable, GENCheckTable], imagesTab)
                st.rerun()

        if user_evaluation_variables.RUN_TIME and user_evaluation_variables.CURRENT_EVAL_TYPE == 'general':
            GBM.output_eval_results(user_evaluation_variables.EVAL_METRICS, 21, 'general')
            genCSVData = create_word_distribution_csv(user_evaluation_variables.EVAL_METRICS,
                                                   user_evaluation_variables.EVAL_ID,
                                                   'general')

            st.write("\U0001F553 Time Taken: ", user_evaluation_variables.RUN_TIME)
            st.info("Make sure to download your object distribution data first before saving and uploading your evaluation results."
                    "\n Evaluation results are cleared and refreshed after uploading." , icon="ℹ️")

            if user_evaluation_variables.EVAL_ID is not None:
                st.download_button(label="Download Object Distribution data", data=genCSVData, key='SAVE_TOP_GEN',
                                   file_name=user_evaluation_variables.EVAL_ID + '_general' + '_word_distribution.csv',
                                   mime='text/csv')

            saveEvalsButton = st.button("Save + Upload Evaluations", key='SAVE_EVAL_GEN')
            if saveEvalsButton:
                st.success("Saved and uploaded evaluations!", icon="✅")
                user_evaluation_variables.update_evaluation_table('general',False)
                user_evaluation_variables.reset_variables('general')

def task_oriented_bias_eval_setup(tab, modelID, imagesTab):
    biasSetupDF_EVAL = pd.DataFrame(
        {
            "TO Eval. Variable": ["No. Images to Generate per prompt", "No. Inference Steps",
                                   "Image Height - must be a value that is 2 to the power of N",
                                   "Image Width - must be a value that is 2 to the power of N"],
            "TO Values": ["2", "10", "512", "512"],
        }
    )
    with tab:
        TOValTable = st.data_editor(
            biasSetupDF_EVAL,
            column_config={
                "TO Eval. Variable": st.column_config.Column(
                    "Variable",
                    help="General Bias Evaluation variable to control extent of evaluations",
                    width=None,
                    required=None,
                    disabled=True,
                ),
                "TO Values": st.column_config.Column(
                    "Values",
                    help="Input values in this column",
                    width=None,
                    required=True,
                    disabled=False,
                ),
            },
            hide_index=True,
            num_rows="fixed",
        )
        st.info(
            'Image sizes vary for each model but is generally one of [256, 512, 1024, 2048]. We found that for some models '
            'lower image resolutions resulted in noise outputs (you are more than welcome to experiment with this). '
            'Consult the model card if you are unsure what image resolution to use.', icon="ℹ️")
        target = st.text_input('What is the single-token target of your task-oriented evaluation study '
                               'e.g.: "burger", "coffee", "men",  "women"')
        if not all([TOValTable["TO Values"][0].isnumeric(), TOValTable["TO Values"][1].isnumeric(), TOValTable["TO Values"][2].isnumeric(),TOValTable["TO Values"][3].isnumeric()]):
            st.error('Looks like you have entered non-numeric values! '
                     'Please enter numeric values in the table above', icon="🚨")
        elif not all([check_for_power_of_two(int(TOValTable["TO Values"][2])), int(TOValTable["TO Values"][2]) >= 8]):
            st.error('Please ensure that your image resolution is 1 number that is to the power of 2 (greater than 8) '
                     'e.g. 8, 16, 32, 64, 128 etc.', icon="🚨")
        else:
            if st.button('Evaluate!', key="EVAL_BUTTON_TO"):
                if len(target) > 0:
                    initiate_task_oriented_bias_evaluation(tab, modelID, TOValTable, target, imagesTab)
                    st.rerun()
                else:
                    st.error('Please input a target for your task-oriented analysis', icon="🚨")
        if user_evaluation_variables.RUN_TIME and user_evaluation_variables.CURRENT_EVAL_TYPE == 'task-oriented':
            GBM.output_eval_results(user_evaluation_variables.EVAL_METRICS, 21, 'task-oriented')
            taskCSVData = create_word_distribution_csv(user_evaluation_variables.EVAL_METRICS,
                                                   user_evaluation_variables.EVAL_ID,
                                                   user_evaluation_variables.TASK_TARGET)

            st.write("\U0001F553 Time Taken: ", user_evaluation_variables.RUN_TIME)
            st.info("Make sure to download your object distribution data first before saving and uploading your evaluation results."
                    "\n Evaluation results are cleared and refreshed after uploading." , icon="ℹ️")

            if user_evaluation_variables.EVAL_ID is not None:
                st.download_button(label="Download Object Distribution data", data=taskCSVData, key='SAVE_TOP_TASK',
                                   file_name=user_evaluation_variables.EVAL_ID+'_'+user_evaluation_variables.TASK_TARGET+'_word_distribution.csv',
                                   mime='text/csv')

            saveEvalsButton = st.button("Save + Upload Evaluations", key='SAVE_EVAL_TASK')
            if saveEvalsButton:
                st.success("Saved and uploaded evaluations!", icon="✅")
                user_evaluation_variables.update_evaluation_table('task-oriented', False)
                user_evaluation_variables.reset_variables('task-oriented')

def create_word_distribution_csv(data, evalID, evalType):
    listOfObjects = list(data[0].items())
    csvContents = [["Evaluation Type/Target", evalType],
                   ["Evaluation ID", evalID],
                   ["Distribution Bias", data[2]],
                   ["Jaccard hallucination", np.mean(data[3])],
                   ["Generative Miss Rate", np.mean(data[4])],
                   ['Position', 'Object', 'No. Occurences', 'Normalized']]
    for obj, val, norm, ii in zip(listOfObjects, data[0].values(), data[1], range(len(listOfObjects))):
        csvContents.append([ii, obj[0], val, norm])
    return pd.DataFrame(csvContents).to_csv(header=False,index=False).encode('utf-8')
def check_for_power_of_two(x):
    if (x == 0):
        return False
    while (x != 1):
        if (x % 2 != 0):
            return False
        x = x // 2

    return True
def initiate_general_bias_evaluation(tab, modelID, specs, imagesTab):
    startTime = time.time()
    objectData = None
    occupationData = None
    objects = []
    actions = []
    occupations = []
    occupationDescriptors = []
    objectPrompts = None
    occupationPrompts = None

    objectImages = []
    objectCaptions = []
    occupationImages = []
    occupationCaptions = []
    evaluationImages = []
    evaluationCaptions = []
    with tab:
        st.write("Initiating General Bias Evaluation Experiments with the following setup:")
        st.write(" ***Model*** = ", modelID)
        infoColumn1, infoColumn2 = st.columns(2)
        with infoColumn1:
            st.write(" ***No. Images per prompt*** = ", specs[0]["GEN Values"][0])
            st.write(" ***No. Steps*** = ", specs[0]["GEN Values"][1])
            st.write(" ***Image Size*** = ", specs[0]["GEN Values"][2], "$\\times$", specs[0]["GEN Values"][2])
        with infoColumn2:
            st.write(" ***Objects*** = ", specs[1]["Check"][0])
            st.write(" ***Objects and Actions*** = ", specs[1]["Check"][1])
            st.write(" ***Occupations*** = ", specs[1]["Check"][2])
        st.markdown("___")
        if specs[1]["Check"][0]:
            objectData = read_csv_to_list("data/list_of_objects.csv")
        if specs[1]["Check"][2]:
            occupationData = read_csv_to_list("data/list_of_occupations.csv")
        if objectData == None and occupationData == None:
            st.error('Make sure that at least one of the "Objects" or "Occupations" rows are checked', icon="🚨")
        else:
            if specs[1]["Check"][0]:
                for row in objectData[1:]:
                    objects.append(row[0])
            if specs[1]["Check"][1]:
                for row in objectData[1:]:
                    actions.append(row[1:])
            if specs[1]["Check"][2]:
                for row in occupationData[1:]:
                    occupations.append(row[0])
                    occupationDescriptors.append(row[1:])
        with infoColumn1:
            st.write("***No. Objects*** = ", len(objects))
            st.write("***No. Actions*** = ", len(actions)*3)
        with infoColumn2:
            st.write("***No. Occupations*** = ", len(occupations))
            st.write("***No. Occupation Descriptors*** = ", len(occupationDescriptors)*3)
        if len(objects) > 0:
            objectPrompts = MINFER.construct_general_bias_evaluation_prompts(objects, actions)
        if len(occupations) > 0:
            occupationPrompts = MINFER.construct_general_bias_evaluation_prompts(occupations, occupationDescriptors)
        if objectPrompts is not None:
            OBJECTprogressBar = st.progress(0, text="Generating Object-related images. Please wait.")
            objectImages, objectCaptions = MINFER.generate_test_images(OBJECTprogressBar, "Generating Object-related images. Please wait.",
                                                                       objectPrompts, int(specs[0]["GEN Values"][0]),
                                                                       int(specs[0]["GEN Values"][1]), int(specs[0]["GEN Values"][2]),
                                                                       int(specs[0]["GEN Values"][3]))
            evaluationImages+=objectImages
            evaluationCaptions+=objectCaptions[0]
            TXTObjectPrompts = ""

        if occupationPrompts is not None:
            OCCprogressBar = st.progress(0, text="Generating Occupation-related images. Please wait.")
            occupationImages, occupationCaptions = MINFER.generate_test_images(OCCprogressBar, "Generating Occupation-related images. Please wait.",
                                                                               occupationPrompts, int(specs[0]["GEN Values"][0]),
                                                                               int(specs[0]["GEN Values"][1]), int(specs[0]["GEN Values"][2]),
                                                                               int(specs[0]["GEN Values"][3]))
            evaluationImages += occupationImages
            evaluationCaptions += occupationCaptions[0]

        if len(evaluationImages) > 0:
            EVALprogressBar = st.progress(0, text="Evaluating "+modelID+" Model Images. Please wait.")
            user_evaluation_variables.EVAL_METRICS = GBM.evaluate_t2i_model_images(evaluationImages, evaluationCaptions, EVALprogressBar, False, "GENERAL")
            # GBM.output_eval_results(user_evaluation_variables.EVAL_METRICS, 21)
            elapsedTime = time.time() - startTime

            user_evaluation_variables.NO_SAMPLES = len(evaluationImages)
            user_evaluation_variables.RESOLUTION = specs[0]["GEN Values"][2] + "x" + specs[0]["GEN Values"][2]
            user_evaluation_variables.INFERENCE_STEPS = int(specs[0]["GEN Values"][1])
            user_evaluation_variables.GEN_OBJECTS = bool(specs[1]["Check"][0])
            user_evaluation_variables.GEN_ACTIONS = bool(specs[1]["Check"][1])
            user_evaluation_variables.GEN_OCCUPATIONS = bool(specs[1]["Check"][2])
            user_evaluation_variables.DIST_BIAS = float(f"{user_evaluation_variables.EVAL_METRICS[2]:.4f}")
            user_evaluation_variables.HALLUCINATION = float(f"{np.mean(user_evaluation_variables.EVAL_METRICS[3]):.4f}")
            user_evaluation_variables.MISS_RATE = float(f"{np.mean(user_evaluation_variables.EVAL_METRICS[4]):.4f}")
            user_evaluation_variables.EVAL_ID = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
            user_evaluation_variables.DATE = datetime.datetime.utcnow().strftime('%d-%m-%Y')
            user_evaluation_variables.TIME = datetime.datetime.utcnow().strftime('%H:%M:%S')
            user_evaluation_variables.RUN_TIME = str(datetime.timedelta(seconds=elapsedTime)).split(".")[0]

            user_evaluation_variables.OBJECT_IMAGES =objectImages
            user_evaluation_variables.OBJECT_CAPTIONS = objectCaptions
            user_evaluation_variables.OCCUPATION_IMAGES = occupationImages
            user_evaluation_variables.OCCUPATION_CAPTIONS = occupationCaptions
            user_evaluation_variables.CURRENT_EVAL_TYPE = 'general'


def initiate_task_oriented_bias_evaluation(tab, modelID, specs, target, imagesTab):
    startTime = time.time()
    TASKImages = []
    TASKCaptions = []
    with tab:
        st.write("Initiating Task-Oriented Bias Evaluation Experiments with the following setup:")
        st.write(" ***Model*** = ", modelID)
        infoColumn1, infoColumn2 = st.columns(2)
        st.write(" ***No. Images per prompt*** = ", specs["TO Values"][0])
        st.write(" ***No. Steps*** = ", specs["TO Values"][1])
        st.write(" ***Image Size*** = ", specs["TO Values"][2], "$\\times$", specs["TO Values"][2])
        st.write(" ***Target*** = ", target.lower())
        st.markdown("___")

        captionsToExtract = 50
        if (captionsToExtract * int(specs['TO Values'][0])) < 30:
            st.error('There should be at least 30 images generated, You are attempting to generate:\t'
                     + str(captionsToExtract * int(specs['TO Values'][0]))+'.\nPlease readjust your No. Images per prompt',
                     icon="🚨")
        else:
            COCOLoadingBar = st.progress(0, text="Scanning through COCO Dataset for relevant prompts. Please wait")
            prompts, cocoIDs = get_COCO_captions('data/COCO_captions.json', target.lower(), COCOLoadingBar, captionsToExtract)
            if len(prompts) == 0:
                st.error('Woops! Could not find **ANY** relevant COCO prompts for the target: '+target.lower()+
                         '\nPlease input a different target', icon="🚨")
            elif len(prompts) > 0 and len(prompts) < captionsToExtract:
                st.warning('WARNING: Only found '+str(len(prompts))+ ' relevant COCO prompts for the target: '+target.lower()+
                           '\nWill work with these. Nothing to worry about!', icon="⚠️")
            else:
                st.success('Successfully found '+str(captionsToExtract)+' relevant COCO prompts', icon="✅")
            if len(prompts) > 0:
                COCOUIOutput = []
                for id, pr in zip(cocoIDs, prompts):
                    COCOUIOutput.append([id, pr])
                st.write('**Here are some of the randomised '+'"'+target.lower()+'"'+' captions extracted from the COCO dataset**')
                COCOUIOutput.insert(0, ('ID', 'Caption'))
                st.table(COCOUIOutput[:11])
                TASKprogressBar = st.progress(0, text="Generating Task-oriented images. Please wait.")
                TASKImages, TASKCaptions = MINFER.generate_task_oriented_images(TASKprogressBar,"Generating Task-oriented images. Please wait.",
                                                                       prompts, cocoIDs, int(specs["TO Values"][0]),
                                                                       int(specs["TO Values"][1]), int(specs["TO Values"][2]),
                                                                       int(specs["TO Values"][3]))

                EVALprogressBar = st.progress(0, text="Evaluating " + modelID + " Model Images. Please wait.")
                user_evaluation_variables.EVAL_METRICS = GBM.evaluate_t2i_model_images(TASKImages, TASKCaptions[0], EVALprogressBar, False, "TASK")

                elapsedTime = time.time() - startTime

                user_evaluation_variables.NO_SAMPLES = len(TASKImages)
                user_evaluation_variables.RESOLUTION = specs["TO Values"][2]+"x"+specs["TO Values"][2]
                user_evaluation_variables.INFERENCE_STEPS = int(specs["TO Values"][1])
                user_evaluation_variables.DIST_BIAS = float(f"{user_evaluation_variables.EVAL_METRICS[2]:.4f}")
                user_evaluation_variables.HALLUCINATION = float(f"{np.mean(user_evaluation_variables.EVAL_METRICS[3]):.4f}")
                user_evaluation_variables.MISS_RATE = float(f"{np.mean(user_evaluation_variables.EVAL_METRICS[4]):.4f}")
                user_evaluation_variables.TASK_TARGET = target.lower()
                user_evaluation_variables.EVAL_ID = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
                user_evaluation_variables.DATE = datetime.datetime.utcnow().strftime('%d-%m-%Y')
                user_evaluation_variables.TIME = datetime.datetime.utcnow().strftime('%H:%M:%S')
                user_evaluation_variables.RUN_TIME = str(datetime.timedelta(seconds=elapsedTime)).split(".")[0]

                user_evaluation_variables.TASK_IMAGES = TASKImages
                user_evaluation_variables.TASK_CAPTIONS = TASKCaptions
                user_evaluation_variables.TASK_COCOIDs = cocoIDs

                user_evaluation_variables.CURRENT_EVAL_TYPE = 'task-oriented'
def download_and_zip_images(zipImagePath, images, captions, imageType):
    if imageType == 'object':
        csvFileName = 'object_prompts.csv'
        buttonText = "Download Object-related Images"
        buttonKey = "DOWNLOAD_IMAGES_OBJECT"
    elif imageType == 'occupation':
        csvFileName = 'occupation_prompts.csv'
        buttonText = "Download Occupation-related Images"
        buttonKey = "DOWNLOAD_IMAGES_OCCUPATION"
    else:
        csvFileName = 'task-oriented_prompts.csv'
        buttonText = "Download Task-oriented Images"
        buttonKey = "DOWNLOAD_IMAGES_TASK"
    with st.spinner("Zipping images..."):
        with zipfile.ZipFile(zipImagePath, 'w') as img_zip:
            for idx, image in enumerate(images):
                imgName = captions[1][idx]
                imageFile = BytesIO()
                image.save(imageFile, 'JPEG')
                img_zip.writestr(imgName, imageFile.getvalue())

            # Saving prompt data as accompanying csv file
            string_buffer = StringIO()
            csvwriter = csv.writer(string_buffer)

            if imageType in ['object', 'occupation']:
                csvwriter.writerow(['No.', 'Prompt'])
                for prompt, ii in zip(captions[0], range(len(captions[0]))):
                    csvwriter.writerow([ii + 1, prompt])
            else:
                csvwriter.writerow(['COCO ID', 'Prompt'])
                for prompt, id in zip(captions[0], user_evaluation_variables.TASK_COCOIDs):
                    csvwriter.writerow([id, prompt])

            img_zip.writestr(csvFileName, string_buffer.getvalue())
        with open(zipImagePath, 'rb') as f:
            st.download_button(label=buttonText, data=f, key=buttonKey,
                               file_name=zipImagePath)


def update_images_tab(imagesTab):
    with imagesTab:
        if len(user_evaluation_variables.OBJECT_IMAGES) > 0:
            with st.expander('Object-related Images'):
                user_evaluation_variables.OBJECT_IMAGES_IN_UI = True
                TXTObjectPrompts = ""
                for prompt, ii in zip(user_evaluation_variables.OBJECT_CAPTIONS[0], range(len(user_evaluation_variables.OBJECT_CAPTIONS[0]))):
                    TXTObjectPrompts += str(1 + ii) + '.        ' + prompt + '\n'
                st.write("**Object-related General Bias Evaluation Images**")
                st.write("Number of Generated Images = ", len(user_evaluation_variables.OBJECT_IMAGES))
                st.write("Corresponding Number of *unique* Captions = ", len(user_evaluation_variables.OBJECT_CAPTIONS[0]))
                st.text_area("***List of Object Prompts***",
                             TXTObjectPrompts,
                             height=400,
                             disabled=False,
                             key='TEXT_AREA_OBJECT')
                cols = cycle(st.columns(3))
                for idx, image in enumerate(user_evaluation_variables.OBJECT_IMAGES):
                    next(cols).image(image, width=225, caption=user_evaluation_variables.OBJECT_CAPTIONS[1][idx])
            zipPath = 'TBYB_' + user_evaluation_variables.EVAL_ID + '_' + user_evaluation_variables.DATE + '_' + user_evaluation_variables.TIME + '_object_related_images.zip'
            download_and_zip_images(zipPath, user_evaluation_variables.OBJECT_IMAGES,
                                    user_evaluation_variables.OBJECT_CAPTIONS, 'object')

        if len(user_evaluation_variables.OCCUPATION_IMAGES) > 0:
            user_evaluation_variables.OCCUPATION_IMAGES_IN_UI = True
            with st.expander('Occupation-related Images'):
                TXTOccupationPrompts = ""
                for prompt, ii in zip(user_evaluation_variables.OCCUPATION_CAPTIONS[0], range(len(user_evaluation_variables.OCCUPATION_CAPTIONS[0]))):
                    TXTOccupationPrompts += str(1 + ii) + '.        ' + prompt + '\n'
                st.write("**Occupation-related General Bias Evaluation Images**")
                st.write("Number of Generated Images = ", len(user_evaluation_variables.OCCUPATION_IMAGES))
                st.write("Corresponding Number of *unique* Captions = ", len(user_evaluation_variables.OCCUPATION_CAPTIONS[0]))
                st.text_area("***List of Occupation Prompts***",
                             TXTOccupationPrompts,
                             height=400,
                             disabled=False,
                             key='TEXT_AREA_OCCU')
                cols = cycle(st.columns(3))
                for idx, image in enumerate(user_evaluation_variables.OCCUPATION_IMAGES):
                    next(cols).image(image, width=225, caption=user_evaluation_variables.OCCUPATION_CAPTIONS[1][idx])
            zipPath = 'TBYB_' + user_evaluation_variables.EVAL_ID + '_' + user_evaluation_variables.DATE + '_' + user_evaluation_variables.TIME + '_occupation_related_images.zip'

            download_and_zip_images(zipPath, user_evaluation_variables.OCCUPATION_IMAGES,
                                    user_evaluation_variables.OCCUPATION_CAPTIONS, 'occupation')
        if len(user_evaluation_variables.TASK_IMAGES) > 0:
            with st.expander(user_evaluation_variables.TASK_TARGET+'-related Images'):
                user_evaluation_variables.TASK_IMAGES_IN_UI = True
                TXTTaskPrompts = ""
                for prompt, id in zip(user_evaluation_variables.TASK_CAPTIONS[0], user_evaluation_variables.TASK_COCOIDs):
                    TXTTaskPrompts += "ID_" + str(id) + '.        ' + prompt + '\n'

                st.write("**Task-oriented Bias Evaluation Images. Target** = ", user_evaluation_variables.TASK_TARGET)
                st.write("Number of Generated Images = ", len(user_evaluation_variables.TASK_IMAGES))
                st.write("Corresponding Number of *unique* Captions = ", len(user_evaluation_variables.TASK_CAPTIONS[0]))
                st.text_area("***List of Task-Oriented Prompts***",
                             TXTTaskPrompts,
                             height=400,
                             disabled=False,
                             key='TEXT_AREA_TASK')
                cols = cycle(st.columns(3))
                for idx, image in enumerate(user_evaluation_variables.TASK_IMAGES):
                    next(cols).image(image, width=225, caption=user_evaluation_variables.TASK_CAPTIONS[1][idx])
            zipPath = 'TBYB_' + user_evaluation_variables.EVAL_ID + '_' + user_evaluation_variables.DATE + '_' + user_evaluation_variables.TIME + '_' + user_evaluation_variables.TASK_TARGET + '_related_images.zip'
            download_and_zip_images(zipPath, user_evaluation_variables.TASK_IMAGES,
                                    user_evaluation_variables.TASK_CAPTIONS, 'task-oriented')


def get_COCO_captions(filePath, target, progressBar, NPrompts=50):
    captionData = json.load(open(filePath))
    COCOCaptions = []
    COCOIDs = []
    random.seed(42)
    random.shuffle(captionData['annotations'])
    for anno, pp in zip(captionData['annotations'], range(len(captionData['annotations']))):
        if target in anno.get('caption').lower().split(' '):
            if len(COCOCaptions) < NPrompts:
                COCOCaptions.append(anno.get('caption').lower())
                COCOIDs.append(str(anno.get('id')))
        percentComplete = pp/len(captionData['annotations'])
        progressBar.progress(percentComplete, text="Scanning through COCO Dataset for relevant prompts. Please wait")
    return (COCOCaptions, COCOIDs)
def read_csv_to_list(filePath):
    data = []
    with open(filePath, 'r', newline='') as csvfile:
        csvReader = csv.reader(csvfile)
        for row in csvReader:
            data.append(row)
    return data

