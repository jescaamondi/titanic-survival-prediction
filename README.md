# Titanic Survival Prediction with LightGBM

An end-to-end machine learning pipeline that predicts whether a Titanic passenger survived, using data cleaning, feature engineering, a LightGBM classifier and feature-importance analysis.

**Result:** 81.6% accuracy on a held-out test set of 179 passengers.

## The data

`titanic/Titanic-Dataset.csv` is the Kaggle Titanic training set: 891 passengers with their ticket class, sex, age, family aboard, fare and port of embarkation.

- **Target:** `Survived` (0 = no, 1 = yes)
- **Missing values:** `Age` (177), `Cabin` (687), `Embarked` (2)

## Pipeline

### 1. Data cleaning

| Step | What it does |
| --- | --- |
| Age imputation | Missing ages are filled with the median age for the passenger's class and sex, for example 3rd-class men or 1st-class women |
| Embarked imputation | The two missing ports are filled with the most common port (S) |
| Cabin handling | Cabin numbers are dropped and replaced by a `HasCabin` flag (1 if a cabin was recorded) |
| Dropped columns | `PassengerId`, `Ticket` and `Name` |

### 2. Feature engineering

| Feature | Description |
| --- | --- |
| `FamilySize` | `SibSp + Parch + 1`, the number of family members aboard including the passenger |
| `IsAlone` | 1 if the passenger travelled without family |
| `Sex` | male → 0, female → 1 |
| `Embarked` | One-hot encoded with `drop_first=True` |

### 3. Modelling

- **Model:** `LGBMClassifier` (gradient-boosted decision trees)
- **Split:** 80% training, 20% testing, `random_state=42`
- **Metrics:** accuracy, precision, recall and F1-score

## Results

| Class | Precision | Recall | F1-score |
| --- | --- | --- | --- |
| Did not survive | 0.85 | 0.84 | 0.84 |
| Survived | 0.77 | 0.78 | 0.78 |
| **Overall accuracy** | | | **0.816** |

For comparison, a simple logistic regression on the same split reaches about 81%, so most of the signal comes from a few strong features rather than from model complexity.

## What drives survival

LightGBM can measure feature importance in two ways, and they tell different stories:

| Rank | By split count (how often a feature is used) | By gain (how much it improves predictions) |
| --- | --- | --- |
| 1 | Fare | **Sex** |
| 2 | Age | Fare |
| 3 | FamilySize | Age |
| 4 | Sex | Pclass |

Fare and age are used in many splits because they are continuous, so the trees can cut them at many points. Sex is used less often but each split on it improves the predictions far more, which matches the "women and children first" evacuation policy.

## Run it yourself

jupyter notebook titanic/titanic_lightgbm.ipynb
jupyter notebook titanic/logistic_regression_baseline.ipynb

```bash
git clone https://github.com/jescaamondi/titanic-survival-prediction.git
cd titanic-survival-prediction
pip install -r requirements.txt
jupyter notebook titanic/titanic_lightgbm.ipynb
```

`titanic_lightgbm.ipynb` is the main model. `logistic_regression_baseline.ipynb` compares label encoding and one-hot encoding with a logistic regression baseline.

`titanic/cleaning.py` runs the cleaning and feature-engineering steps on their own and writes `titanic/titanic_cleaned.csv`.

## Tools

Python · pandas · NumPy · scikit-learn · LightGBM · Matplotlib · Jupyter
