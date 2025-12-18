import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from src.transformer import FeatureTransformer


# ---- Load and split ----

df = pd.read_csv("data/raw/titanic.csv")

traget = "Survived"

drop_cols = ["Survived", "PassengerId"]

X = df.drop(columns=drop_cols)
y = df["Survived"]


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y)


# ---- Build full pipeline (Use FeatureTransformer) ----

full_pipe = Pipeline([
    ("features", FeatureTransformer(task="classification")),
    ("model", LogisticRegression(max_iter=1000))
]) 

full_pipe.fit(X_train, y_train)
pred = full_pipe.predict(X_test)

accur = accuracy_score(y_test, pred)

print(accur)
print(classification_report(y_test, pred))


# ---- Serialize entire pipeline ----
import joblib
import os

os.makedirs("artifacts", exist_ok=True)

joblib.dump(full_pipe, "artifacts/titanic_pipeline.joblib")

loaded_pipe = joblib.load("artifacts/titanic_pipeline.joblib")

np.testing.assert_array_equal(
    full_pipe.predict(X_test),
    loaded_pipe.predict(X_test)
)

# ---- Real inference on new raw data ----
new_passengers = pd.DataFrame([
    {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S",
        "Name": "John Doe",
        "Ticket": "A/5 21171",
        "Cabin": None
    },
    {
        "Pclass": 1,
        "Sex": "female",
        "Age": 38,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 71.2833,
        "Embarked": "C",
        "Name": "Jane Doe",
        "Ticket": "PC 17599",
        "Cabin": "C85"
    }
])

predictions = loaded_pipe.predict(new_passengers)
probs = loaded_pipe.predict_proba(new_passengers)[:, 1]

for i, (p, pr) in enumerate(zip(predictions, probs)):
    print(f"Passenger {i}: survived={bool(p)}, probability={pr:.2f}")


