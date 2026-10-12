import pandas as pd

def generate_maintenance_report(predictions_df, threshold=0.85):
    
    high_risk = predictions_df[predictions_df['failure_probability'] >= threshold]
    
    if high_risk.empty:
        print("✅ All machines are operating within safe parameters. No imminent failures detected.")
        return
        
    print("\n⚠️ URGENT MAINTENANCE ALERTS DETECTED ⚠️")
    print("=" * 60)
    for _, row in high_risk.iterrows():
        print(f"Machine ID      : {row['machineID']}")
        print(f"Timestamp       : {row['datetime']}")
        print(f"Failure Risk    : {row['failure_probability']:.1%}")
        
        # Diagnostic Insights: Errors in last 24h
        recent_errors = []
        for i in range(1, 6):
            col = f"error{i}_count_24h"
            if col in row and row[col] > 0:
                recent_errors.append(f"error{i} ({int(row[col])}x)")
        err_str = ", ".join(recent_errors) if recent_errors else "None"
        print(f"Recent Warnings : {err_str}")
        # Diagnostic Insights: Component Wear
        wear_info = []
        for c in range(1, 5):
            col = f"days_since_comp{c}"
            if col in row:
                wear_info.append(f"Comp{c}: {row[col]:.0f}d")
        print(f"Component Age   : {', '.join(wear_info)}")
        print("-" * 60)