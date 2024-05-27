# Try Before you Bias (TBYB)

![quantifying_bias](https://github.com/JJ-Vice/TBYB/blob/main/assets/QuantifyingBias.png)

Bias in text-to-image (T2I) models can propagate unfair social representations and may be used to aggressively market ideas or push controversial agendas. Existing T2I model bias evaluation methods only focus on social biases. We look beyond that and instead propose an evaluation methodology to quantify general biases in T2I generative models, without any preconceived notions. We assess four state-of-the-art T2I models and compare their baseline bias characteristics to their respective variants (two for each), where certain biases have been intentionally induced. We propose three evaluation metrics to assess model biases including: (i) Distribution bias, (ii) Jaccard hallucination and (iii) Generative miss-rate. We conduct two evaluation studies, modelling biases under general, and task-oriented conditions, using a marketing scenario as the domain for the latter. We also quantify social biases to compare our findings to related works. Finally, our methodology is transferred to evaluate captioned-image datasets and measure their bias. Our approach is objective, domain-agnostic and consistently measures different forms of T2I model biases. 

This repository contains all code and data necessary to replicate experiments in the paper:

J. Vice, N. Akhtar, R. Hartley and A. Mian, "Quantifying Bias in Text-to-Image Generative Models,"  _arXiv preprint_, arXiv:2312.13053, 2023.

Available: https://arxiv.org/abs/2312.13053

A hosted version of the application (with comparisons with other user evaluations) is available as a [HuggingFace Space](https://huggingface.co/spaces/JVice/try-before-you-bias), currently running on an NVIDIA T4 GPI. A video series with demonstrations is available on [YouTube](https://www.youtube.com/watch?v=3pKWilbPjzU). 

While the HuggingFace space is more regularly maintained and allows you to store and compare your evaluations to other evaluations, this version allows for a local clone that can be experimented on.

# Installation
The TBYB application was designed using the [Streamlit](https://streamlit.io/) Python UI framework. The application was tested locally using a single NVIDIA GeForce RTX 4090. However, when running locally, the computational requirements are dependent on the model imported for analysis. The larger the model, the more computational resources required.
 - Streamlit (V. 1.28.1)
 - Python (V. 3.9.16)


A compatible `conda` virtual environment is recommended and a full list of package dependencies is available in **requirements.txt**. Once you have cloned the repository, setup the environment with
```
$ conda create -n tbyb python=3.9.16
$ conda activate tbyb
$ pip install -r requirements.txt
```

# Usage
Run the TBYB application using the command
```shell
streamlit run main.py
```
This will assign a local and network URL's to view the application e.g.:
```shell
Local URL: http://localhost:8501
Network URL: http://00.111.222.333:8501
```
## Citation
If our code, metrics or paper are used to further your research, please cite our paper:
```BibTeX
@misc{Vice2023Quantifying,
      title={Quantifying Bias in Text-to-Image Generative Models}, 
      author={Jordan Vice and Naveed Akhtar and Richard Hartley and Ajmal Mian},
      year={2023},
      eprint={2312.13053},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Constraints
While we have attempted to design a comprehensive, automated bias evaluation tool. We must acknowledge that in its infancy, TBYB has some constraints:
- We have not checked the validity of *every* single T2I model and model type on HuggingFace so we cannot promise that all T2I models will work - if you run into any issues that you think should be possible, feel free to reach out!
- Currently, a model_index.json file is required to load models and use them with TBYB, we will look to address other models in the future.
- Target models must be public. Gated models are currently not supported.
- TBYB only works on T2I models hosted on HuggingFace, other model repositories are not currently supported.
- Adaptor models are currently not supported, we will look to add evaluation functionalities of these models in the future.
- BLIP and CLIP models used for evaluations are limited by their own biases and object recognition capabilities. However, manual evaluations of bias could result in subjective labelling biases.   


# Disclaimer 
Given this application is used for the assessment of T2I biases and relies on pre-trained models available on HuggingFace, we are not responsible for any content generated by public-facing models that have been used to generate images using this application. 
Bias cannot be easily measured and we do not claim that our approach is without any faults. TBYB is proposed as an auxiliary tool to assess model biases and thus, if a chosen model is found to output insensitive, disturbing, distressing or offensive images that propagate harmful stereotypes or representations of marginalised groups, please address your concerns to the model providers.

However, given the TBYB tool is designed for bias quantification and is driven by transparency, it would be beneficial to the TBYB community to share evaluations of biased T2I models!

Despite only being able to assess HuggingFace \U0001F917 models, we share no association with them outside of hosting TBYB as a HuggingFace space. Given their growth in popularity in the computer science community and their host of T2I model repositories, we have decided to host our web-app here.

For further help, please refer to the "Additional Information" and "How to Use" tabs of the application.

If you have any questions/queries or if you want to simply strike a conversation, please reach out to Jordan Vice at: jordan.vice@uwa.edu.au or raise an issue in the [Issues tab](https://github.com/JJ-Vice/TBYB/issues)
