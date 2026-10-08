from src.data_processing import load_data, engineer_features, create_labels
from src.model import split_data, train_xgboost, evaluate_model
from src.reporting import generate_maintenance_report

def main():
    print("1. Loading dataset...")
    telemetry, failures, machines = load_data()

    print("2. Engineering Features...")
    features_df = engineer_features(telemetry, machines)

    print("3. Creating Labels for Training...")
    data_df = create_labels(features_df, failures, horizon_hours= 24)

    #drop non feature columns
    features = [c for c in data_df.columns if c not in ['machineID', 'datetime', 'failure', 'isfailure', 'target_24h', 'model']]

    print("4. Splitting data...")
    train, test = split_data(data_df, train_fraction= 0.8)
    X_train, y_train = train[features], train['target_24h']
    X_test, y_test = test[features], test['target_24h']

    print("5. Training XGBoost Model...")
    model = train_xgboost(X_train, y_train)

    print("6. Evaluating Model...")
    evaluate_model(model, X_test, y_test)

    print("\n7. Generating Sample Maintenance Report (from latest test records)...")
    latest_records = test.groupby('machineID').tail(1).copy()
    latest_probs = model.predict_proba(latest_records[features])[:, 1]
    latest_records['failure_probability'] = latest_probs
    
    generate_maintenance_report(latest_records, threshold=0.85)
    

if __name__ == "__main__":
    main()