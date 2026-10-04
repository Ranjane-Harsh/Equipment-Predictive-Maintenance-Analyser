import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
import joblib

def split_data(df, train_fraction = 0.8):
    split_idx = int(len(df) * train_fraction)
    df_sorted = df.sortvalues('datetime')

    train = df_sorted.iloc[:split_idx]
    test = df_sorted.iloc[split_idx:]

    return train, test