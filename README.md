# BRF-web-tool

The objective of this research was to develop and validate a precise and scalable machine learning model for predicting in-hospital mortality among patients with acute myocardial infarction (AMI) in intensive care units (ICUs). Additionally, we aimed to assess the significance of each feature in the model using SHapley Additive exPlanations (SHAP) values.

To obtain the source data, access is required to [MIMIC-III](https://mimic.mit.edu/docs/iii/), [MIMIC-IV](https://mimic.mit.edu/docs/iv/), and [eICU](https://eicu-crd.mit.edu/)). Please refer to our paper for detailed data preprocessing steps, wherein we specifically extracted patients diagnosed with AMI for the first time after admission to the ICU.

Subsequently, you can utilize the code in this repository to build the model, analyze the data, and generate all the charts mentioned in the paper.

You can execute app_streamlit.py to set up an application service, enabling direct utilization of pre-trained tools (model parameters will be made public after the paper's publication).
