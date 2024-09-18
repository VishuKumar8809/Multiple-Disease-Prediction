# Multiple-Disease-Prediction
Here’s a README file template for your Multiple Disease Prediction project using Streamlit and a saved model in `.sav` format:

---

# Multiple Disease Prediction System

This project implements a Multiple Disease Prediction System using machine learning models saved in `.sav` format. The application is built using Streamlit for the user interface, allowing users to predict the likelihood of various diseases by providing input features such as medical history and clinical measurements.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Streamlit Run Command](#streamlit-run-command)
- [Future Enhancements](#future-enhancements)
- [References](#references)

## Project Overview

The Multiple Disease Prediction System predicts the probability of several common diseases (e.g., diabetes, heart disease, cancer) based on user inputs. The machine learning models for each disease are trained offline, and the trained models are stored in `.sav` format using Python's Pickle module. The system provides a simple web-based interface through Streamlit, allowing users to interact with the model in real time.

## Features
- User-friendly interface for disease prediction.
- Predicts the probability of multiple diseases from user inputs.
- Uses pre-trained machine learning models in `.sav` format.
- Easy to deploy and run via Streamlit.

## System Architecture

1. Input: The user inputs health-related data (e.g., age, blood pressure, glucose levels) through the Streamlit web interface.
2. Model: The system loads the pre-trained machine learning models from `.sav` files using the `pickle` module.
3. Prediction: The model predicts the probability of various diseases based on the input features.
4. Output: The system displays the prediction results (e.g., "Diabetes: High Risk" or "Heart Disease: Low Risk").

## Installation

### Prerequisites
Make sure you have Python 3 installed on your machine. You'll also need the following libraries:
- Streamlit
- Scikit-learn
- Pickle
- Streamlit Option Menu

### Steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/multiple-disease-prediction.git
    cd multiple-disease-prediction
    ```

2. Install the required libraries:
    ```bash
    pip install streamlit scikit-learn streamlit-option-menu
    ```

3. Download or place your saved models:
   Make sure you have the `.sav` files of your trained models in the `models` directory.

4. File Structure:
    ```
    multiple-disease-prediction/
    ├── models/
    │   ├── diabetes_model.sav
    │   ├── heart_disease_model.sav
    │   └── cancer_model.sav
    ├── app.py
    ├── README.md
    └── requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Select Disease for Prediction:
    - The app will show a sidebar menu using Streamlit Option Menu. Users can select from multiple diseases (e.g., diabetes, heart disease).
    - Input the required health parameters, such as age, glucose levels, and blood pressure.

3. View the Results:
    - The app will display the disease prediction results, indicating whether the user is at high or low risk for the selected disease.

### Sample Code

```python
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Load models
diabetes_model = pickle.load(open('models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('models/heart_disease_model.sav', 'rb'))

# Sidebar menu for disease selection
with st.sidebar:
    selected = option_menu('Disease Prediction',
                           ['Diabetes Prediction', 'Heart Disease Prediction'],
                           icons=['activity', 'heart'],
                           default_index=0)

# Diabetes Prediction
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction')
    
    # Input fields
    age = st.number_input('Age', min_value=1, max_value=120, step=1)
    glucose = st.number_input('Glucose Level', min_value=0)
    
    # Prediction button
    if st.button('Predict'):
        prediction = diabetes_model.predict([[age, glucose]])[0]
        if prediction == 1:
            st.error('High Risk of Diabetes')
        else:
            st.success('Low Risk of Diabetes')

# Heart Disease Prediction
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction')
    
    # Input fields
    age = st.number_input('Age', min_value=1, max_value=120, step=1)
    blood_pressure = st.number_input('Blood Pressure', min_value=0)
    
    # Prediction button
    if st.button('Predict'):
        prediction = heart_disease_model.predict([[age, blood_pressure]])[0]
        if prediction == 1:
            st.error('High Risk of Heart Disease')
        else:
            st.success('Low Risk of Heart Disease')
```

## Streamlit Run Command

To run the application using Streamlit, use the following command:

```bash
streamlit run app.py
```

Ensure that you have installed the necessary dependencies by following the steps under the [Installation](#installation) section.

## Future Enhancements
- Add more diseases and corresponding trained models for prediction.
- Improve the user interface with more interactive data inputs and visualizations.
- Deploy the application using services like Heroku or AWS for wider accessibility.
- Integrate a database to store user input and results for future reference.

## References
- Streamlit Documentation: [https://docs.streamlit.io/](https://docs.streamlit.io/)
- Scikit-learn Documentation: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- Pickle Module: [https://docs.python.org/3/library/pickle.html](https://docs.python.org/3/library/pickle.html)

---

This README provides a comprehensive overview of the project setup, usage, and instructions for running the Streamlit application with the saved machine learning models.
