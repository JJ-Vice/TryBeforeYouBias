import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
import model_comparison as MCOMP
import model_loading as MLOAD
import model_inferencing as MINFER
import user_evaluation_variables
import tab_manager
import yaml
from yaml.loader import SafeLoader
from PIL import Image

TBYB_LOGO = Image.open('./assets/TBYB_logo_light.png')

def setup_page_banner():
    global USER_LOGGED_IN
    # for tab in [tab1, tab2, tab3, tab4, tab5]:
    c1,c2,c3,c4,c5,c6,c7,c8,c9 = st.columns(9)
    with c5:
        st.image(TBYB_LOGO, use_column_width=True)
    for col in [c1,c2,c3,c4,c5,c6,c7,c8,c9]:
        col = None
    st.title('Try Before You Bias (TBYB)')
    st.write('*A Quantitative T2I Bias Evaluation Tool*')
def setup_how_to():
    expander = st.expander("How to Use")
    expander.write("1. Navigate to the '\U0001F527 Setup' tab and input the ID of the HuggingFace \U0001F917 T2I model you want to evaluate\n")
    expander.image(Image.open('./assets/HF_MODEL_ID_EXAMPLE.png'))
    expander.write("2. Test your chosen model by generating an image using an input prompt e.g.: 'A corgi with some cool sunglasses'\n")
    expander.image(Image.open('./assets/lykon_corgi.png'))
    expander.write("3. Navigate to the '\U0001F30E Bias Evaluation (BEval)' or '\U0001F3AF Task-Oriented BEval' tabs "
                   "   to evaluate your model once it has been loaded\n"
                   "4. Once you have generated some evaluation images, head over to the '\U0001F4C1 Generated Images' tab to have a look at them\n"
                   "5. To check out your evaluations or all of the TBYB Community evaluations, head over to the '\U0001F4CA Model Comparison' tab\n"
                   "6. For more information about the evaluation process, see our paper at https://arxiv.org/abs/2312.13053 or navigate to the "
                   "   '\U0001F4F0 Additional Information' tab for a TL;DR.\n"
                   "7. For any questions or to report any bugs/issues. Please contact jordan.vice@uwa.edu.au.\n")

def setup_additional_information_tab(tab):
    with tab:
        st.header("1. Quantifying Bias in Text-to-Image (T2I) Generative Models")
        st.markdown(
            """
            *Based on the article of the same name available here --PAPER HYPERLINK--
            
            Authors: Jordan Vice, Naveed Akhtar, Richard Hartley and Ajmal Mian
            
            This web-app was developed by **Jordan Vice** to accompany the article, serving as a practical 
            implementation of how T2I model biases can be quantitatively assessed and compared. Evaluation results from 
            all *base* models discussed in the paper have been incorporated into the TBYB community results and we hope 
            that others share their evaluations as we look to further the discussion on transparency and reliability 
            of T2I models. 
            
            """)

        st.header('2. A (very) Brief Summary')
        st.image(Image.open('./assets/TBYB_flowchart.png'))
        st.markdown(
                    """
                    Bias in text-to-image models can propagate unfair social representations and could be exploited to 
                    aggressively market ideas or push controversial or sinister agendas. Existing T2I model bias evaluation
                    methods focused on social biases. So, we proposed a bias evaluation methodology that considered
                    general and task-oriented biases, spawning the Try Before You Bias (**TBYB**) application as a result.
                    """
                )
        st.markdown(
        """
            We proposed three novel metrics to quantify T2I model biases:
            1. Distribution Bias - $B_D$
            2. Jaccard Hallucination - $H_J$
            3. Generative Miss Rate - $M_G$
            
            Open the appropriate drop-down menu to understand the logic and inspiration behind metric.
            """
        )
        c1,c2,c3 = st.columns(3)
        with c1:
            with st.expander("Distribution Bias - $B_D$"):
                st.markdown(
                    """
                    Using the Area under the Curve (AuC) as an evaluation metric in machine learning is not novel. However,
                    in the context of T2I models, using AuC allows us to define the distribution of objects that have been
                    detected in generated output image scenes.
                    
                    So, everytime an object is detected in a scene, we update a dictionary (which is available for
                    download after running an evaluation). After evaluating a full set of images, you can use this 
                    information to determine what objects appear more frequently than others. 
                    
                    After all images are evaluated, we sort the objects in descending order and normalize the data. We
                    then use the normalized values to calculate $B_D$, using the trapezoidal AuC rule i.e.:
                    
                    $B_D = \\Sigma_{i=1}^M\\frac{n_i+n_{i=1}}{2}$ 
                    
                    So, if a user conducts a task-oriented study on biases related to **dogs** using a model
                    that was heavily biased using pictures of animals in the wild. You might find that after running
                    evaluations, the most common objects detected were trees and grass - even if these objects weren't 
                    specified in the prompt. This would result in a very low $B_D$ in comparison to a model that for
                    example was trained on images of dogs and animals in various different scenarios $\\rightarrow$
                    which would result in a *higher* $B_D$ in comparison.
                    """
                )
        with c2:
            with st.expander("Jaccard Hallucination - $H_J$"):
                st.markdown(
                    """
                    Hallucination is a very common phenomena that is discussed in relation to generative AI, particularly
                    in relation to some of the most popular large language models. Depending on where you look, hallucinations
                    can be defined as being positive, negative, or just something to observe $\\rightarrow$ a sentiment
                    that we echo in our bias evaluations.
                    
                    Now, how does hallucination tie into bias? In our work, we use hallucination to define how often a
                    T2I model will *add* objects that weren't specified OR, how often it will *omit* objects that were
                    specified. This indicates that there could be an innate shift in bias in the model, causing it to 
                    add or omit certain objects. 
                    
                    Initially, we considered using two variables $H^+$ and $H^-$ to define these two dimensions of 
                    hallucination. Then, we considered the Jaccard similarity coefficient, which
                    measures the similarity *and* diversity of two sets of objects/samples - defining this as 
                    Jaccard Hallucination - $H_J$. 
                    
                    Simply put, we define the set of objects detected in the input prompt and then detect the objects in 
                    the corresponding output image. Then, we determine the intersect over union. For a model, we
                    calculate the average $H_J$ across generated images using:
                    
                    $H_J = \\frac{\Sigma_{i=0}^{N-1}1-\\frac{\mathcal{X}_i\cap\mathcal{Y}_i}{\mathcal{X}_i\cup\mathcal{Y}_i}}{N}$

                    """
                )
        with c3:
            with st.expander("Generative Miss Rate - $M_G$"):
                st.markdown(
                    """
                    Whenever fairness and trust are discussed in the context of machine learning and AI systems, 
                    performance is always highlighted as a key metric - regardless of the downstream task. So, in terms
                    of evaluating bias, we thought that it would be important to see if there was a correlation 
                    between bias and performance (as we predicted). And while the other metrics do evaluate biases
                    in terms of misalignment, they do not consider the relationship between bias and performance.
                    
                    We use an additional CLIP model to assist in calculating Generative Miss Rate - $M_G$. Logically,
                    as a model becomes more biased, it will begin to diverge away from the intended target and so, the
                    miss rate of the generative model will increase as a result. This was a major consideration when
                    designing this metric.
                    
                    We use the CLIP model as a binary classifier, differentiating between two classes:
                    - the prompt used to generate the image
                    - **NOT** the prompt 
                    
                    Through our experiments on intentionally-biased T2I models, we found that there was a clear 
                    relationship between $M_G$ and the extent of bias. So, we can use this metric to quantify and infer 
                    how badly model performances have been affected by their biases. 
                    """
                )
        st.header('3. TBYB Constraints')
        st.markdown(
            """
            While we have attempted to design a comprehensive, automated bias evaluation tool. We must acknowledge that
            in its infancy, TBYB has some constraints:
            - We have not checked the validity of *every* single T2I model and model type on HuggingFace so we cannot 
            promise that all T2I models will work - if you run into any issues that you think should be possible, feel
            free to reach out!
            - Currently, a model_index.json file is required to load models and use them with TBYB, we will look to
            address other models in future works
            - TBYB only works on T2I models hosted on HuggingFace, other model repositories are not currently supported
            - Adaptor models are not currently supported, we will look to add evaluation functionalities of these
            models in the future.
            - BLIP and CLIP models used for evaluations are limited by their own biases and object recognition
            capabilities. However, manual evaluations of bias could result in subjective labelling biases.                        
            - Download, generation, inference and evaluation times are all hardware dependent. 
            
            Keep in mind that these constraints may be removed or added to any time.
            """)
        st.header('4. Disclaimer')
        st.markdown(
            """
            Given this application is used for the assessment of T2I biases and relies on 
            pre-trained models available on HuggingFace, we are not responsible for any content generated 
            by public-facing models that have been used to generate images using this application. 
            
            Bias cannot be easily measured and we do not claim that our approach is without any faults. TBYB is proposed 
            as an auxiliary tool to assess model biases and thus, if a chosen model is found to output
            insensitive, disturbing, distressing or offensive images that propagate harmful stereotypes or
            representations of marginalised groups, please address your concerns to the model providers.

            However, given the TBYB tool is designed for bias quantification and is driven by transparency, it would be 
            beneficial to the TBYB community to share evaluations of biased T2I models!

            Despite only being able to assess HuggingFace \U0001F917 models, we share no association with them outside of
            hosting TBYB as a HuggingFace space. Given their growth in popularity in the computer science community and their
            host of T2I model repositories, we have decided to host our web-app here.

            For further questions/queries or if you want to simply strike a conversation, 
            please reach out to Jordan Vice at: jordan.vice@uwa.edu.au""")

setup_page_banner()
setup_how_to()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["\U0001F527 Setup", "\U0001F30E Bias Evaluation (BEval)", "\U0001F3AF Task-Oriented BEval",
                                       "\U0001F4CA Model Comparison", "\U0001F4C1 Generated Images", "\U0001F4F0 Additional Information"])
setup_additional_information_tab(tab6)

# PLASTER THE LOGO EVERYWHERE
tab2.subheader("General Bias Evaluation")
tab2.write("Waiting for \U0001F527 Setup to be complete...")
tab3.subheader("Task-Oriented Bias Evaluation")
tab3.write("Waiting for \U0001F527 Setup to be complete...")
tab4.write("Check out other model evaluation results from users across the **TBYB** Community! \U0001F30E ")
tab4.write("You can also just compare your own model evaluations by clicking the '*Personal Evaluation*' buttons")
MCOMP.initialise_page(tab4)
tab5.subheader("Generated Images from General and Task-Oriented Bias Evaluations")
tab5.write("Waiting for \U0001F527 Setup to be complete...")

with tab1:
    with st.form("model_definition_form", clear_on_submit=True):
        modelID = st.text_input('Input the HuggingFace \U0001F917 T2I model_id for the model you '
                                'want to analyse e.g.: "runwayml/stable-diffusion-v1-5"')
        submitted1 = st.form_submit_button("Submit")
        if modelID:
            with st.spinner('Checking if ' + modelID + ' is valid and downloading it (if required)'):
                modelLoaded = MLOAD.check_if_model_exists(modelID)
                if modelLoaded is not None:
                    # st.write("Located " + modelID + " model_index.json file")
                    st.write("Located " + modelID)

                    modelType = MLOAD.get_model_info(modelLoaded)
                    if modelType is not None:
                        st.write("Model is of Type: ", modelType)

                        if submitted1:
                            MINFER.TargetModel = MLOAD.import_model(modelID, modelType)
                            if MINFER.TargetModel is not None:
                                st.write("Text-to-image pipeline looks like this:")
                                st.write(MINFER.TargetModel)
                                user_evaluation_variables.MODEL = modelID
                                user_evaluation_variables.MODEL_TYPE = modelType
                else:
                    st.error('The Model: ' + modelID + ' does not appear to exist or the model does not contain a model_index.json file.'
                                                       ' Please check that that HuggingFace repo ID is valid.'
                                                       ' For more help, please see the "How to Use" Tab above.', icon="🚨")
                    modelID = None
    if modelID:
        with st.form("example_image_gen_form", clear_on_submit=True):
            testPrompt = st.text_input('Input a random test prompt to test out your '
                                       'chosen model and see if its generating images:')
            submitted2 = st.form_submit_button("Submit")
            if testPrompt and submitted2:
                with st.spinner("Generating an image with the prompt:\n"+testPrompt+"(This may take some time)"):
                    testImage = MINFER.generate_test_image(MINFER.TargetModel, testPrompt)
                st.image(testImage, caption='Model: ' + modelID + ' Prompt: ' + testPrompt)
                st.write('''If you are happy with this model, navigate to the other tabs to evaluate bias!
                              Otherwise, feel free to load up a different model and run it again''')

    if MINFER.TargetModel is not None:
        tab_manager.completed_setup([tab2, tab3, tab4, tab5], modelID)
# else:
#     MCOMP.databaseDF = None
#     user_evaluation_variables.reset_variables('general')
#     user_evaluation_variables.reset_variables('task-oriented')
