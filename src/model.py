import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
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

    scale_weight = neg_cases / pos_cases if pos_cases > 0 else 1.0

    model = XGBClassifier(
        n_estimators = 100,
        max_depth = 5,
        learning_rate = 0.1,
        scale_pos_weight = scale_weight,
        random_state = 42
    )

    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)

    print(f"ROC-AUC Score: {auc:.4f}")
    print("\nClassification Report: ")
    print(classification_report(y_test, preds))
    
    return auc