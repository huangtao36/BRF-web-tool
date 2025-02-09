# BRF-web-tool

![image](https://github.com/huangtao36/BRF-web-tool/data/215942_359.png)
![image](https://github.com/huangtao36/BRF-web-tool/data/220052_265.png)

The image shows the results and explanations of a machine learning model used to predict in-hospital mortality rates. The model predicts an in-hospital mortality probability of 10.5%, which is an increase of 89.5% compared to the baseline probability.

Explanation of the image:
- Live Probability Scores: 10.5: Indicates that the model predicts a 10.5% probability of in-hospital mortality.
- ↑ 89.5: Indicates an increase of 89.5% in the predicted mortality probability compared to the baseline.
- SHAP Waterfall Plot: Shows the contributions of each feature to the prediction outcome.
- f(x) = 0.895: The final predicted probability of in-hospital mortality for the input data x is 89.5%.
- E[f(X)] = 0.673: The average predicted probability across all data is 67.3%.

Feature Contributions:
- APSIII: Contributes +0.13, the largest positive contribution.
- Surgery: Contributes +0.04.
- SysBP (Systolic Blood Pressure): Contributes -0.04, the largest negative contribution.
- eGFR (Estimated Glomerular Filtration Rate): Contributes +0.03.
- HR (Heart Rate): Contributes -0.03.
- RR (Respiratory Rate): Contributes +0.02.
- Haemoglobin: Contributes -0.02.
- Age: Contributes +0.02.
- AG (possibly a medical indicator such as anion gap): Contributes +0.02.
- Other features: Unspecified features contribute a total of +0.04.

These contributions indicate how each feature affects the final predicted probability. Positive values increase the predicted probability, while negative values decrease it. This visualization helps to understand how the model makes predictions and which features have the most significant impact on the outcome.

---

The objective of this research was to develop and validate a precise and scalable machine learning model for predicting in-hospital mortality among critically ill patients with acute myocardial infarction (AMI). Additionally, we aimed to assess the significance of each feature in the model using SHapley Additive exPlanations (SHAP) values.

To obtain the source data, access is required to [MIMIC-III](https://mimic.mit.edu/docs/iii/), [MIMIC-IV](https://mimic.mit.edu/docs/iv/), and [eICU](https://eicu-crd.mit.edu/). Please refer to our paper for detailed data preprocessing steps, wherein we specifically extracted patients diagnosed with AMI for the first time after admission to the ICUs.

Subsequently, you can utilize the code in this repository to build the model, analyze the data, and generate all the charts mentioned in the paper.

You can execute `app_streamlit.py` to set up an application service, enabling direct utilization of pre-trained tools (model parameters will be made public after the paper's publication).

Alternatively, you can try out the tool we've developed directly at [http://115.29.201.138:1234/](http://115.29.201.138:1234/). Feel free to provide any suggestions. (If you encounter any accessibility issues, please let me know.)
