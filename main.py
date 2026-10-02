from src.data_processing import load_data

def main():
    telemetry, failures, machines = load_data()

if __name__ == "__main__":
    main()