import pandas as pd

def generate_maintenance_report(predictions_df, threshold=0.85):
    
    high_risk = predictions_df[predictions_df['failure_probability'] >= threshold]
    
    if high_risk.empty:
        print("✅ All machines are operating within safe parameters. No imminent failures detected.")
        return
        
    print("⚠️ URGENT MAINTENANCE ALERTS ⚠️")
    print("=" * 40)
    for _, row in high_risk.iterrows():
        print(f"Machine ID: {row['machineID']}")
        print(f"Timestamp : {row['datetime']}")
        print(f"Risk Prob : {row['failure_probability']:.1%}")
        print("-" * 40)