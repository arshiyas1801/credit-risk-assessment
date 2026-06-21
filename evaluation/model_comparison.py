import numpy as np
import pandas as pd

results={
    "Models" : ["Neural network","random forest","XGBoost"],
    "Accuracy": [0.735, 0.755, 0.775],
    "Precision": [0.7458, 0.7791, 0.7987],
    "Recall": [0.9429, 0.9071, 0.9071],
    "F1 Score": [0.8328, 0.8383, 0.8495]
}

comparison_df=pd.DataFrame(results)

comparison_df=comparison_df.round(3)

print(comparison_df)

# Save for Streamlit
comparison_df.to_csv("evaluation/model_comparison.csv", index=False)