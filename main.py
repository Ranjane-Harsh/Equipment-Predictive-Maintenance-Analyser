from src.data_processing import load_data, engineer_features, create_labels
from src.model import split_data

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
    
    

if __name__ == "__main__":
    main()