# Diabetes Predictor

## Overview
The **Diabetes Predictor** project uses machine learning to predict whether a person is diabetic based on input features. The model is built using the **Decision Tree Classifier** and tuned using hyperparameter optimization techniques such as **GridSearchCV** and **RandomizedSearchCV**.

## Dataset
The dataset used for this project is `DATA1.csv`, which contains various health-related features used to predict diabetes. 

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

To install the required packages, you can use the following:

```bash
pip install -r requirements.txt

Diabetes-Predictor/
│
├── DATA1.csv               # The dataset used for training and testing
├── diabetes_predictor.py   # Python script containing the code
├── requirements.txt        # List of required Python packages
└── README.md               # This README file
