import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from utils import preprocess, RANDOM_STATE, TEST_SIZE

# prepare data for modeling
def prepare_data(df):
    df_long = df.reset_index().melt(id_vars=['Time (s)'], var_name='Voltage', value_name='Temperature') #converts wide-format data to long-format.
    df_long['Voltage'] = df_long['Voltage'].astype(float)
    return df_long.dropna()

# train and evaluate model
def train_and_evaluate(data, label):
    results = []
    combined_y_test, combined_y_pred = [], []

    for voltage in data['Voltage'].unique():
        voltage_data = data[data['Voltage'] == voltage]
        if len(voltage_data) < 2:
            continue

        X = voltage_data[['Time (s)', 'Voltage']]
        y = voltage_data['Temperature']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        results.append({
            "Voltage (V)": voltage,
            "Mean Squared Error": mse,
            "Root Mean Squared Error": rmse,
            "R2 Score": r2,
            "Mean Absolute Percentage Error (%)": mape
        })

        combined_y_test.extend(y_test)
        combined_y_pred.extend(y_pred)

    # print results
    results_df = pd.DataFrame(results)
    print(f"\nSummary Table for {label}:")
    print(results_df.to_string(index=False))

    # plot Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(combined_y_test, combined_y_pred, alpha=0.7, label="Observed")
    plt.plot([min(combined_y_test), max(combined_y_test)], [min(combined_y_test), max(combined_y_test)], 'r--', label="Ideal Fit")
    plt.xlabel("Actual Temperature")
    plt.ylabel("Predicted Temperature")
    plt.title(f"Predicted vs Actual Temperatures for {label}")
    plt.legend()
    plt.grid(True)
    plt.show()

# define file paths for datasets
file_paths = {
    'Smooth Laminar': 'smooth_data_with_times.csv',
    'Smooth Turbulent': 'smooth_turbulent_with_times.csv',
    'Microstructure Laminar': 'micro_data_with_times.csv',
    'Microstructure Turbulent': 'micro_turbulent_with_times.csv'
}

# process data
datasets = preprocess(file_paths)

# train and evaluate models
for label, dataset in datasets.items():
    prepared_data = prepare_data(dataset)
    train_and_evaluate(prepared_data, label)
