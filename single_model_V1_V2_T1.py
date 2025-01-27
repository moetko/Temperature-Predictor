import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from utils import preprocess, RANDOM_STATE, TEST_SIZE


# prepare data with V1, V2, T1, and T2
def prepare_data_v1_v2_t1_t2(df):
    # transform to long format 
    df = df.reset_index().melt(id_vars=['Time (s)'], var_name='Voltage', value_name='Temperature')
    # convert 'Voltage' to numeric
    df['Voltage'] = df['Voltage'].astype(float)
    df = df.sort_index()
    df['V1'] = df['Voltage']
    df['V2'] = df['Voltage'].shift(-1)
    df['T1'] = df['Temperature']
    df['T2'] = df['Temperature'].shift(-1)
    return df[['V1', 'V2', 'T1', 'T2']].dropna()

def train_and_evaluate_v1_v2_t1_t2(data_prepared, label):
    # input features and target
    X = data_prepared[['V1', 'V2', 'T1']]
    y = data_prepared['T2']

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # train model
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # predict and evaluate
    y_pred = model.predict(X_test)

    # overall metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # print overall metrics
    print(f"\nEvaluation Metrics for {label}:")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Root Mean Squared Error: {rmse:.6f}")
    print(f"R2 Score: {r2:.6f}")
    print(f"Mean Absolute Percentage Error: {mape:.6f}%")

    # metrics grouped by voltage
    results = []
    combined_y_test, combined_y_pred = [], []
    for voltage in data_prepared['V1'].unique():
        voltage_mask = X_test['V1'] == voltage
        if voltage_mask.sum() == 0:
            continue  # skip if no test samples for this voltage
        
        y_test_voltage = y_test[voltage_mask]
        y_pred_voltage = y_pred[voltage_mask]

        voltage_mse = mean_squared_error(y_test_voltage, y_pred_voltage)
        voltage_rmse = sqrt(voltage_mse)
        voltage_r2 = r2_score(y_test_voltage, y_pred_voltage)
        voltage_mape = np.mean(np.abs((y_test_voltage - y_pred_voltage) / y_test_voltage)) * 100

        results.append({
            "Voltage (V)": voltage,
            "Mean Squared Error": voltage_mse,
            "Root Mean Squared Error": voltage_rmse,
            "R2 Score": voltage_r2,
            "Mean Absolute Percentage Error (%)": voltage_mape
        })

        combined_y_test.extend(y_test_voltage)
        combined_y_pred.extend(y_pred_voltage)

    # convert results to DataFrame and print
    results_df = pd.DataFrame(results)
    print(f"\nMetrics by Voltage for {label}:")
    print(results_df.to_string(index=False))

    # plot Actual vs Predicted 
    plt.figure(figsize=(8, 6))
    plt.scatter(combined_y_test, combined_y_pred, alpha=0.7, label="Observed")
    plt.plot([min(combined_y_test), max(combined_y_test)], [min(combined_y_test), max(combined_y_test)], 'r--', lw=2, label="Ideal Fit")
    plt.xlabel("Actual Temperature")
    plt.ylabel("Predicted Temperature")
    plt.title(f"Predicted vs Actual Temperatures for {label}")
    plt.legend()
    plt.grid(True)
    plt.show()

# define file paths for datasets
file_paths = {
    "Smooth Laminar": "smooth_data_with_times.csv",
    "Microstructure Laminar": "micro_data_with_times.csv",
    "Smooth Turbulent": "smooth_turbulent_with_times.csv",
    "Microstructure Turbulent": "micro_turbulent_with_times.csv"
}

# preprocess datasets
datasets = preprocess(file_paths)

# prepare, train, and evaluate each dataset
for label, df in datasets.items():
    prepared_data = prepare_data_v1_v2_t1_t2(df)
    train_and_evaluate_v1_v2_t1_t2(prepared_data, label)
