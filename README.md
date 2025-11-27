#  AI-Protein-Predictor: Stability Classification Pipeline

FOCUS: Applying Machine Learning to solve early-stage protein engineering challenges, specifically filtering unstable candidates.

##  Pipeline Summary
* **Methodology:** Trains a **Random Forest Classifier (Scikit-learn)** using features derived from protein sequences.
* **Feature Engineering:** Utilizes **Amino Acid Composition (AAC)** as input features (Pandas/NumPy).
* **Analytical Insight:** Model importance identified **Histidine (H), Arginine (R), and Glycine (G)** as the most influential amino acids for stability prediction in the test cohort.
* **Value:** Demonstrates an end-to-end ML workflow from raw biological data to actionable analytical insight.

##  Usage
To run the prototype:
1.  Install dependencies: `pip install pandas scikit-learn`
2.  Run the script: `python stability_predictor.py`
