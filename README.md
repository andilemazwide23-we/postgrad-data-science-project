# Stroke Prediction Using Machine Learning

Binary classification project predicting stroke risk from patient health and demographic data. 
Completed as part of the STAT606 coursework in the Postgraduate Diploma in Data Science (University of KwaZulu-Natal),Group of 2.

Final Grade : **84%**

## Project Overview

Stroke is a leading cause of death and disability worldwide. Early identification of high-risk patients is critical for prevention.
This project builds and compares several machine learning models to predict whether a patient will experience a stroke (Yes/No) based on medical and demographic factors.

## Dataset

- **Source:** Healthcare Stroke Prediction dataset(Kaggle)
- **Size:** 5,109 records
- **Target variable:** `stroke` (1 = Stroke, 0 = No Stroke)
- **Class distribution:** Stroke 4.87% | No Stroke 95.13% (severe imbalance)
- **Features used:** Age, hypertension, heart disease, average glucose level, BMI, smoking status, residence type, marital status

## Data Preparation

- Dropped the ID variable (not predictive)
- Imputed missing BMI values using the median
- One-hot encoded categorical variables
- Removed the rare "Other" gender category
- Dropped the redundant age-group variable and low-relevance work-type variable from the final model

## Handling Class Imbalance

Given the severe imbalance (4.87% stroke cases), several resampling techniques were evaluated using logistic regression as a baseline: under-sampling, over-sampling, combined sampling, and SMOTE. **SMOTE** was selected based on strong AUC performance, balancing the training set to 50/50 while leaving the test set unchanged for realistic evaluation.

## Models

Four classification techniques were trained on the SMOTE-balanced training set (70:30 train/test split) and evaluated on the untouched test set:

- Naive Bayes
- Decision Tree (H2O, tuned — max_depth = 13, min_rows = 1)
- Decision Tree (rpart)
- Logistic Regression

## Results

| Model | Train AUC | Test AUC | AUC Gap | Diagnosis |
|---|---|---|---|---|
| Naive Bayes | 0.8319 | 0.8123 | 0.0196 | Good generalisation |
| Decision Tree (H2O) | 0.9886 | 0.6659 | 0.3223 | Severe overfitting — high variance |
| Decision Tree (rpart) | 0.7685 | 0.7346 | 0.0339 | Moderate gap |
| **Logistic Regression** | **0.8535** | **0.8323** | **0.0212** | **Best bias-variance trade-off** |

**Best model: Logistic Regression** — selected using AUC (a threshold-independent metric well suited to imbalanced classification), achieving the highest test AUC (0.83) and the strongest MCC among the four models. The H2O decision tree showed clear overfitting, with a large gap between training and test performance. All models achieved high recall, though precision remained low across the board due to the class imbalance.

### Top 5 Predictors (Logistic Regression, standardised coefficients)

| Variable | Std. Coefficient | Odds Ratio | P-value |
|---|---|---|---|
| Age | 1.7841 | 1.0838 | <0.0001 |
| Hypertension | 0.4661 | 1.5937 | 0.0001 |
| Heart disease | 0.3462 | 1.4137 | 0.0143 |
| Average glucose level | 0.2377 | 1.0043 | <0.0001 |
| Formerly smoked | 0.1546 | 1.1672 | 0.1497 |

Age is by far the strongest predictor — each additional year increases the odds of stroke by roughly 8%. Hypertension is the second strongest predictor.

## Conclusion

Logistic regression combined with SMOTE resampling provided the most reliable and interpretable baseline for stroke prediction, outperforming more complex tree-based methods on this dataset. Age, hypertension, and heart disease emerged as the most important clinical predictors.

## Repository Contents

- `stroke_prediction_model.R` — R script containing data cleaning, modelling, and evaluation code
- `stroke prediction presentation.pdf` — presentation slides
- `Presentation Recording.mp4` — 10-minute video presentation of the project

## Tools

R, with packages for SMOTE resampling, H2O (decision tree), rpart (decision tree), and logistic regression modelling.

## My Contribution

This was a collaborative project completed by a team of two students. My contributions included:

##Data preprocessing and cleaning
 Exploratory Data Analysis (EDA)
 Building and evaluating machine learning models
 Interpreting model performance
 Preparing the final presentation and documentation

