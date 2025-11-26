# stability_predictor.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 20 standard amino acids
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

def calculate_aac(sequence):
    """
    Calculates the Amino Acid Composition (AAC) for a protein sequence.
    Returns the percentage of each amino acid in the sequence.
    """
    # Convert sequence to uppercase and remove whitespace
    sequence = sequence.upper().replace(" ", "")
    seq_len = len(sequence)
    
    # Return zero percentages if sequence is empty
    if seq_len == 0:
        return {aa: 0.0 for aa in AMINO_ACIDS}

    # Initialize dictionary to store AA counts
    aac_features = {}
    
    for aa in AMINO_ACIDS:
        # Calculate count and percentage
        count = sequence.count(aa)
        percentage = (count / seq_len) * 100
        aac_features[aa] = round(percentage, 4) # 4 decimal places precision
        
    return aac_features

# ---------------------------------------------------------------------
# STEP 2: DATA SIMULATION AND PANDAS STRUCTURING
# ---------------------------------------------------------------------

# Sample dataset (Sequences and Stability Labels)
# 1=Stable, 0=Unstable
data = {
    'Sequence': [
        "AALGLGALAGV", 
        "CDEWIPSSKLM", 
        "HIKLLTSSLLM", 
        "QWERTYUIOPQ",
        "STABLEGHKL",
        "UNSTABLEC",
        "GPYYAAACDEF",
        "KKRRLLMNNP"
    ],
    'Stability_Label': [1, 0, 1, 0, 1, 0, 1, 0] 
}

# Create DataFrame
df = pd.DataFrame(data)

# Apply AAC function to the 'Sequence' column and expand results into new columns
aac_data = df['Sequence'].apply(calculate_aac).apply(pd.Series)

# Concatenate original DataFrame with AAC features
df = pd.concat([df, aac_data], axis=1)

print("\n--- Step 2 Result: Pandas DataFrame (First 5 Rows) ---")
print(df.head())
print("---------------------------------------------------\n")
# stability_predictor.py (Continuation)

# -----------------------------------------------------
# STEP 3: MODEL TRAINING AND EVALUATION (Scikit-learn)
# -----------------------------------------------------

# 1. Separate Features (X) and Target (Y)
# X: All AAC columns (features)
# Y: Stability_Label (target)

# We drop 'Sequence' and 'Stability_Label' from X to keep only the AAC features
X = df.drop(['Sequence', 'Stability_Label'], axis=1)
Y = df['Stability_Label']

# 2. Split Data into Training and Test Sets
# We use a small dataset split for demonstration purposes.
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

print(f"Total Samples: {len(X)} | Training Samples: {len(X_train)} | Test Samples: {len(X_test)}")

# 3. Initialize and Train the Model
# We use a robust classifier (Random Forest) to handle the features.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train) # Model eğitimi burada gerçekleşir

# 4. Predict and Evaluate
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)


print("\n" + "="*50)
print(" MODEL EVALUATION SUMMARY (CV OUTPUT)")
print("="*50)
print(f" Prediction Accuracy (Test Set): {accuracy:.4f}")
print("-----------------------------------------------------")
# stability_predictor.py (Continuation: Feature Importance)

# -----------------------------------------------------
# STEP 4: FEATURE IMPORTANCE (Analytical Proof)
# -----------------------------------------------------

# Random Forest modelinin özellik önemlerini alıyoruz
importances = model.feature_importances_
feature_names = X.columns
forest_importances = pd.Series(importances, index=feature_names)

# En önemli ilk 5 özelliği bulalım
top_n = 5
top_features = forest_importances.nlargest(top_n)

print("\n" + "="*50)
print(f" TOP {top_n} FEATURES FOR STABILITY PREDICTION")
print("="*50)
print(top_features)
