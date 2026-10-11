from src.data_processing import load_data, engineer_features, create_labels
from src.model import split_data, train_xgboost, evaluate_model
from src.reporting import generate_maintenance_report

def main():
    print("1. Loading dataset...")
    telemetry, failures, machines, errors, maint = load_data()

    print("2. Engineering Features...")
    features_df = engineer_features(telemetry, machines, errors, maint)

    print(features_df.head(20). to_string())
    


    """print("3. Creating Labels for Training...")
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
    optimal_threshold = evaluate_model(model, X_test, y_test)
    
    #save_model(model, "xgboost_pdm.joblib")
    
    print("\n7. Generating Diagnostic Maintenance Report...")
    # Extract the most recent reading for each machine in the test set
    latest_records = test.groupby('machineID').tail(1).copy()
    
    # Compute probabilities using the trained model
    latest_probs = model.predict_proba(latest_records[features])[:, 1]
    latest_records['failure_probability'] = latest_probs
    
    # UPDATE: Use the calibrated optimal_threshold instead of a hardcoded 0.85
    print(f"Applying calibrated threshold of {optimal_threshold:.2%} to filter alerts:")
    generate_maintenance_report(latest_records, threshold=optimal_threshold)"""
    

if __name__ == "__main__":
    main()