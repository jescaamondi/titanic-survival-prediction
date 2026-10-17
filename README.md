Titanic Survival Prediction with LightGBM
A complete end-to-end machine learning pipeline for predicting passenger survival on the Titanic using LightGBM, featuring advanced data cleaning, feature engineering, and model interpretability.

Overview
This project builds a lightGBM model to predict whether a Titanic passenger survived based on demographic and ticket information.
Dataset

    Source: Titanic-Dataset.csv (Kaggle Titanic competition dataset)
    Target: Survived (0 = No, 1 = Yes)
    Features: Passenger demographics, ticket class, family size, fare, port of embarkation, and cabin availability

Pipeline

1. Data Cleaning
Step	Description
Age Imputation	Missing ages filled with the median age per Pclass × Sex group (e.g., 1st-class female median, 3rd-class male median)
Embarked Imputation	Missing ports filled with the overall mode (S)
Cabin Handling	Original cabin numbers dropped; binary HasCabin flag created (1 if cabin was recorded)
Drop Irrelevant	PassengerId, Ticket, and Name removed to reduce noise

2. Feature Engineering
Feature	Description
FamilySize	SibSp + Parch + 1 (total family members aboard)
IsAlone	Binary flag: 1 if FamilySize == 1, else 0
Sex	Mapped to binary: male → 0, female → 1
Embarked	One-hot encoded with drop_first=True to avoid multicollinearity
3. Modeling

    Algorithm: LightGBMClassifier (gradient-boosted decision trees)
    Train/Test Split: 80/20 with random_state=42
    Evaluation Metrics: Accuracy, Precision, Recall, F1-score

4. Model Interpretability
Two feature importance analyses are performed:

    Split Importance: How frequently a feature is used to split nodes
    Gain Importance: How much each feature contributes to reducing the loss function
