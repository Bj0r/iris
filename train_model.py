# train_model.py

import os
import joblib          
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "Iris.csv")       
MODEL_PATH = os.path.join(BASE_DIR, "iris_model.pkl")  


print(f"[train_model] Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)


X = df.drop(columns=["Id", "Species"])  
y = df["Species"]                        

print(f"[train_model] Features : {list(X.columns)}")
print(f"[train_model] Samples  : {len(X)}")


model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)
print("[train_model] Training complete.")


joblib.dump(model, MODEL_PATH)
print(f"[train_model] Model saved to : {MODEL_PATH}")
