import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve, average_precision_score
import joblib

def split_data(df, train_fraction = 0.8):
    split_idx = int(len(df) * train_fraction)
    df_sorted = df.sort_values('datetime')

    train = df_sorted.iloc[:split_idx]
    test = df_sorted.iloc[split_idx:]

    return train, test

def train_xgboost(X_train, y_train):

    neg_cases = (y_train == 0).sum()
    pos_cases = (y_train == 1).sum()

    scale_weight = np.sqrt(neg_cases / pos_cases) 

    model = XGBClassifier(
        n_estimators = 100,
        max_depth = 5,
        learning_rate = 0.1,
        scale_pos_weight = scale_weight,
        random_state = 42
    )

    model.fit(X_train, y_train)
    
    return model

def find_optimal_threshold(y_true, y_probs):
    """
    Finds the probability threshold that maximizes the F1-Score.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Calculate F1 score for every possible threshold
    # (avoid division by zero with eps)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"Optimal Threshold: {best_threshold:.4f} (Yields F1-Score: {best_f1:.4f})")
    return best_threshold

def evaluate_model(model, X_test, y_test):
    # Predict probabilities (NOT discrete 0/1 classes)
    probs = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"PR-AUC Score (True metric for imbalance): {pr_auc:.4f}")
    
    # 2. Find and apply the calibrated threshold
    optimal_thresh = find_optimal_threshold(y_test, probs)
    custom_preds = (probs >= optimal_thresh).astype(int)
    
    print("\nClassification Report (At Optimal Threshold):")
    print(classification_report(y_test, custom_preds))
    
    return optimal_thresh