
import pandas as pd
import numpy as np

df= pd.read_csv("Titanic-Dataset.csv")

print("Initial shape:", df.shape)
print(df.isnull().sum())

df["Age"]= df.groupby(["Pclass", "Sex"])["Age"].transform(
    lambda x: x.fillna(x.median())
)

df["Embarked"]= df["Embarked"].fillna(df["Embarked"].mode()[0])

df["HasCabin"]= df["Cabin"].notnull().astype(int)
df.drop(columns=["Cabin", "PassengerId", "Ticket", "Name"], inplace=True)

df["FamilySize"]= df["SibSp"] + df["Parch"] + 1
df["IsAlone"]= (df["FamilySize"] == 1).astype(int)

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

print("\nFinal shape:", df.shape)
print(df.isnull().sum())
print(df.head())

df.to_csv("titanic_cleaned.csv", index=False)
print("\nSaved cleaned dataset as titanic_clean.csv")
