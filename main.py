from src.data_processing import load_data, engineer_features, create_labels

def main():
    telemetry, failures, machines = load_data()
    features_df = engineer_features(telemetry, machines)
    data_df = create_labels(features_df, failures, horizon_hours= 24)

    print(data_df)
    

if __name__ == "__main__":
    main()