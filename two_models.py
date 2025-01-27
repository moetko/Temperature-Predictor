import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from sklearn.cluster import KMeans
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from utils import preprocess, WINDOW_SIZE, RANDOM_STATE, TEST_SIZE


# calculate rolling statistics
def calculate_rolling_stats(data, columns):
    for column in columns:
        data[f'{column}_Rolling_Mean'] = data[column].rolling(WINDOW_SIZE).mean().fillna(0)
        data[f'{column}_Rolling_Std'] = data[column].rolling(WINDOW_SIZE).std().fillna(0)
        data[f'{column}_Rolling_Var'] = data[column].rolling(WINDOW_SIZE).var().fillna(0)
    return data


# add lagged features
def add_lagged_features(data, target_column='Temperature', max_lag=3):
    for lag in range(1, max_lag + 1):
        data[f'{target_column}_Lag_{lag}'] = data[target_column].shift(lag).fillna(0)
    return data

# prepare data
def prepare_data(df):
    df = df.reset_index().melt(id_vars=['Time (s)'], var_name='Voltage', value_name='Temperature')
    df['Voltage'] = df['Voltage'].astype(float)
    df = calculate_rolling_stats(df, ['Temperature'])
    df = df[np.abs(zscore(df['Temperature'])) < 3].dropna()
    return df

# compute boiling threshold
def compute_boiling_threshold(data):
    data['Rolling_Temp_Var'] = data['Temperature'].rolling(5).var().fillna(0)
    clustering_data = data[['Voltage', 'Rolling_Temp_Var']].dropna()
    kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE)
    clusters = kmeans.fit_predict(clustering_data)
    clustering_data['Cluster'] = clusters
    boiling_cluster = clusters.max()
    return clustering_data.loc[clustering_data['Cluster'] == boiling_cluster, 'Voltage'].min() or data['Voltage'].quantile(0.9)

# classify and split data into phases
def classify_and_split(data):
    threshold = compute_boiling_threshold(data)
    return data[data['Voltage'] < threshold], data[data['Voltage'] >= threshold]

# train and evaluate model
def train_and_evaluate_model(data, label):
    data['Interaction_Term'] = data['Time (s)'] * data['Voltage']
    feature_columns = [
        'Time (s)', 'Voltage',
        'Temperature_Rolling_Mean', 'Temperature_Rolling_Std', 'Temperature_Rolling_Var'
    ] + [col for col in data.columns if 'Lag' in col]

    X = data[feature_columns]
    y = data['Temperature']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        "MSE": mse,
        "R2": r2_score(y_test, y_pred),
        'RMSE': sqrt(mse),
        "MAPE (%)": np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }

    print(f"\n{label} Metrics: {metrics}")

    # plot results
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.7, label="Observed")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Ideal Fit")
    plt.xlabel("Actual Temperature")
    plt.ylabel("Predicted Temperature")
    plt.title(f"Predicted vs Actual Temperatures for {label}")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, metrics

# compute metrics by voltage level
def compute_metrics(data, model, feature_columns):
    data['Predicted_Temperature'] = model.predict(data[feature_columns])
    metrics = []

    for voltage in data['Voltage'].unique():
        voltage_data = data[data['Voltage'] == voltage]
        if len(voltage_data) > 1:
            y_true = voltage_data['Temperature']
            y_pred = voltage_data['Predicted_Temperature']
            mse = mean_squared_error(y_true, y_pred)
            metrics.append({
                "Voltage": voltage,
                "MSE": mse,
                "RMSE": sqrt(mse),
                "R2": r2_score(y_true, y_pred),
                "MAPE (%)": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            })

    return pd.DataFrame(metrics)

# process and train with voltage metrics
def process_and_train_with_voltage_metrics(data, label):
    pre_boiling, boiling = classify_and_split(data)
    print(f"\nTraining models for {label}...")

    # train for pre-boiling phase
    pre_model, pre_metrics = train_and_evaluate_model(pre_boiling, f"{label} (Pre-Boiling)")
    pre_voltage_metrics = compute_metrics(pre_boiling, pre_model, feature_columns=[
        'Time (s)', 'Voltage',
        'Temperature_Rolling_Mean', 'Temperature_Rolling_Std', 'Temperature_Rolling_Var'
    ] + [col for col in pre_boiling.columns if 'Lag' in col])

    # train for boiling phase
    boil_model, boil_metrics = train_and_evaluate_model(boiling, f"{label} (Boiling)")
    boil_voltage_metrics = compute_metrics(boiling, boil_model, feature_columns=[
        'Time (s)', 'Voltage',
        'Temperature_Rolling_Mean', 'Temperature_Rolling_Std', 'Temperature_Rolling_Var'
    ] + [col for col in boiling.columns if 'Lag' in col])

    return {
        "Pre-Boiling": {"Model": pre_model, "Metrics": pre_metrics, "Voltage Metrics": pre_voltage_metrics},
        "Boiling": {"Model": boil_model, "Metrics": boil_metrics, "Voltage Metrics": boil_voltage_metrics}
    }

file_paths = {
    "Smooth Laminar": 'smooth_data_with_times.csv',
    "Microstructure Laminar": 'micro_data_with_times.csv',
    "Smooth Turbulent": 'smooth_turbulent_with_times.csv',
    "Microstructure Turbulent": 'micro_turbulent_with_times.csv'
}

results = {}

# preprocess datasets
datasets = preprocess(file_paths)

# train and evaluate models
results = {}
for label, df in datasets.items():
    # prepare the data for the specific label
    prepared_data = prepare_data(df)
    prepared_data = add_lagged_features(prepared_data)  # add lagged features
    
    # train and evaluate model
    results[label] = process_and_train_with_voltage_metrics(prepared_data, label)

# summarize results
summary = []
voltage_metrics_summary = []

for label, res in results.items():
    for phase, info in res.items():
        summary.append({"Dataset": label, "Phase": phase, **info["Metrics"]})
        phase_voltage_metrics = info["Voltage Metrics"]
        phase_voltage_metrics["Dataset"] = label
        phase_voltage_metrics["Phase"] = phase
        voltage_metrics_summary.append(phase_voltage_metrics)

summary_df = pd.DataFrame(summary)
voltage_metrics_summary_df = pd.concat(voltage_metrics_summary, ignore_index=True)

# print results
print("\nSummary of Results:")
print(summary_df)

print("\nMetrics by Voltage Levels:")
print(voltage_metrics_summary_df)
# print all rows of the DataFrame in the console
print(voltage_metrics_summary_df.to_string())
