# train_model.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE : Train a Decision Tree classifier on the Iris dataset and save
#           the trained model to disk as iris_model.pkl using joblib.
#           Run this script ONCE before launching app.py.
#
# HOW TO RUN:
#   $ python train_model.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import joblib          # used to serialize (save) and deserialize (load) the model
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# ── 1. PATHS ─────────────────────────────────────────────────────────────────
# BASE_DIR   → the folder where this script lives
# DATA_PATH  → Iris.csv must be in the same folder as this script
# MODEL_PATH → the trained model will be saved here as a .pkl binary file
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "Iris.csv")        # input  : raw dataset
MODEL_PATH = os.path.join(BASE_DIR, "iris_model.pkl")  # output : serialized model

# ── 2. LOAD DATA ─────────────────────────────────────────────────────────────
# Read the Iris CSV into a pandas DataFrame.
# Expected columns: Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species
print(f"[train_model] Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# ── 3. PREPARE FEATURES & TARGET ─────────────────────────────────────────────
# X → feature matrix  : drop 'Id' (not useful) and 'Species' (this is what we predict)
# y → target labels   : the Species column (Iris-setosa / Iris-versicolor / Iris-virginica)
X = df.drop(columns=["Id", "Species"])  # shape: 150 rows × 4 feature columns
y = df["Species"]                        # shape: 150 labels

print(f"[train_model] Features : {list(X.columns)}")
print(f"[train_model] Samples  : {len(X)}")

# ── 4. TRAIN THE MODEL ────────────────────────────────────────────────────────
# DecisionTreeClassifier learns a set of if/else rules from the training data.
# random_state=42 ensures the same tree is built every time (reproducibility).
# .fit(X, y) trains the model — it finds the best splits across all 4 features.
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)
print("[train_model] Training complete.")

# ── 5. SAVE THE TRAINED MODEL ─────────────────────────────────────────────────
# joblib.dump() serializes the fitted model object into a binary .pkl file.
# app.py will later call joblib.load() to restore it — no retraining needed.
joblib.dump(model, MODEL_PATH)
print(f"[train_model] Model saved to : {MODEL_PATH}")