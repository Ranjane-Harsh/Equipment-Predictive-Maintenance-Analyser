import pandas as pd
import numpy as np

def load_data(data_dir = "data"):
    telemetry = pd.read_csv(f"{data_dir}/PdM_telemetry.csv")
    failures = pd.read_csv(f"{data_dir}/PdM_failures.csv")
    machines = pd.read_csv(f"{data_dir}/PdM_machines.csv")

    telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])
    failures['datetime'] = pd.to_datetime(failures['datetime'])

    print(telemetry, failures, machines)

    return telemetry, failures, machines