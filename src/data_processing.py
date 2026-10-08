import pandas as pd
import numpy as np

def load_data(data_dir = "data"):
    telemetry = pd.read_csv(f"{data_dir}/PdM_telemetry.csv")
    failures = pd.read_csv(f"{data_dir}/PdM_failures.csv")
    machines = pd.read_csv(f"{data_dir}/PdM_machines.csv")

    telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])
    failures['datetime'] = pd.to_datetime(failures['datetime'])

    return telemetry, failures, machines

def engineer_features(telemetry, machines):
    telemetry = telemetry.sort_values(by=['machineID','datetime'])

    cols = ['volt', 'rotate', 'pressure', 'vibration']

    roll_3h = telemetry.groupby('machineID')[cols].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    roll_3h.columns = [f"{c}_3h_mean" for c in cols] 

    roll_24h = telemetry.groupby('machineID')[cols].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    roll_24h.columns = [f"{c}_24h_mean" for c in cols]

    features = pd.concat([telemetry, roll_3h, roll_24h], axis=1)

    features = features.merge(machines, on='machineID', how='left')

    return features

def create_labels(features, failures, horizon_hours = 24):
    features = features.copy()
    features = features.merge(failures, on=['machineID', 'datetime'], how='left')
    features['failure'] = features['failure'].fillna('none')
    features['isfailure'] = (features['failure'] != 'none').astype(int)

    features['target_24h'] = (
        features.groupby('machineID')['isfailure']
        .transform(lambda s: s.bfill(limit=horizon_hours))
    )
    features['target_24h'] = features['target_24h'].fillna(0).astype(int)

    return features