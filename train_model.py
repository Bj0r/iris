# train_model.py
import os
import joblib         
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score           

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "Iris.csv")       
MODEL_PATH = os.path.join(BASE_DIR, "iris_model.pkl") 

print(f"[train_model] Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Id", "Species"])  
y = df["Species"]                       

print(f"[train_model] Features : {list(X.columns)}")
print(f"[train_model] Samples  : {len(X)}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"[train_model] Training samples : {len(X_train)}")
print(f"[train_model] Testing samples  : {len(X_test)}")


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("[train_model] Training complete.")


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"[train_model] Model Accuracy : {accuracy * 100:.2f}%")


joblib.dump(model, MODEL_PATH)
print(f"[train_model] Model saved to : {MODEL_PATH}")