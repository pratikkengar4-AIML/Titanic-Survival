import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("/kaggle/input/titanic/train.csv")

df.head()

df.info()

df.isnull().sum()

df.describe()

sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival Based on Gender")
plt.show()

sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


df.drop(["Cabin","Name","Ticket","PassengerId"], axis=1, inplace=True)
df.head()


le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

df.head()


X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42
)


lr = LogisticRegression()
lr.fit(X_train,y_train)

y_pred_lr = lr.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))


dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train,y_train)

y_pred_dt = dt.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test,y_pred_dt))


rf = RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train,y_train)

y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test,y_pred_rf))


cm = confusion_matrix(y_test,y_pred_rf)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


print(classification_report(y_test,y_pred_rf))

