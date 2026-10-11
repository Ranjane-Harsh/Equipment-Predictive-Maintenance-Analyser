import pandas as pd
import numpy as np

def load_data(data_dir = "data"):
    telemetry = pd.read_csv(f"{data_dir}/PdM_telemetry.csv")
    failures = pd.read_csv(f"{data_dir}/PdM_failures.csv")
    machines = pd.read_csv(f"{data_dir}/PdM_machines.csv")
    errors = pd.read_csv(f"{data_dir}/PdM_errors.csv")
    maint = pd.read_csv(f"{data_dir}/PdM_maint.csv")

    telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])
    failures['datetime'] = pd.to_datetime(failures['datetime'])
    errors['datetime'] = pd.to_datetime(errors['datetime'])
    maint['datetime'] = pd.to_datetime(maint['datetime'])

    return telemetry, failures, machines, errors, maint

def engineer_features(telemetry, machines, errors, maint):
    telemetry = telemetry.sort_values(by=['machineID','datetime'])

    cols = ['volt', 'rotate', 'pressure', 'vibration']

    # 3h rolling mean
    roll_3h = telemetry.groupby('machineID')[cols].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    roll_3h.columns = [f"{c}_3h_mean" for c in cols] 

    # 24h rolling mean
    roll_24h = telemetry.groupby('machineID')[cols].rolling(window=24, min_periods=1).mean().reset_index(0, drop=True)
    roll_24h.columns = [f"{c}_24h_mean" for c in cols]

    # 24h rolling std dev
    roll_24h_std = telemetry.groupby('machineID')[cols].rolling(window = 24, min_periods = 1).std().reset_index(0, drop=True)
    roll_24h_std.columns = [f"{c}_24h_std" for c in cols]

    # Error counts for past 24 hours
    error_dummies = pd.get_dummies(errors['errorID'], dtype=int)
    errors_encoded = pd.concat([errors[['datetime', 'machineID']], error_dummies], axis=1)
    errors_grouped = errors_encoded.groupby(['datetime', 'machineID']).sum().reset_index()

    telemetry_err = telemetry.merge(errors_grouped, on=['datetime', 'machineID'], how='left')
    error_cols = ['error1', 'error2', 'error3', 'error4', 'error5']
    for c in error_cols:
        if c not in telemetry_err.columns:
            telemetry_err[c] = 0
        else:
            telemetry_err[c] = telemetry_err[c].fillna(0).astype(int)
    roll_errors = telemetry_err.groupby('machineID')[error_cols].rolling(window=24, min_periods=1).sum().reset_index(0, drop=True)
    roll_errors.columns = [f"{c}_count_24h" for c in error_cols]

    # Days elapsed since component replacement
    telemetry_sorted_time = telemetry.sort_values(by='datetime').reset_index(drop=True)
    maint_feats = pd.DataFrame(index=telemetry_sorted_time.index)
    maint_feats['machineID'] = telemetry_sorted_time['machineID']
    maint_feats['datetime'] = telemetry_sorted_time['datetime']
    for comp in ['comp1', 'comp2', 'comp3', 'comp4']:
        m_comp = maint[maint['comp'] == comp].sort_values('datetime').reset_index(drop=True)
        m_comp['maint_date'] = m_comp['datetime']
        merged = pd.merge_asof(
            telemetry_sorted_time[['datetime', 'machineID']],
            m_comp[['maint_date', 'machineID']],
            left_on='datetime',
            right_on='maint_date',
            by='machineID',
            direction='backward'
        )
        maint_feats[f'days_since_{comp}'] = (merged['datetime'] - merged['maint_date']).dt.total_seconds() / (24 * 3600)
    # Re-align maintenance features to the machine-ordered telemetry
    maint_aligned = telemetry[['datetime', 'machineID']].merge(maint_feats, on=['datetime', 'machineID'], how='left')

    # Concatinating all feature blocks
    features = pd.concat([telemetry, roll_3h, roll_24h, roll_24h_std, roll_errors], axis=1)
    for comp in ['comp1', 'comp2', 'comp3', 'comp4']:
        features[f'days_since_{comp}'] = maint_aligned[f'days_since_{comp}']

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