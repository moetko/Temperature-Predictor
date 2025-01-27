# Boiling Heat Transfer Analysis Using Machine Learning
This project applies machine learning algorithms to analyze and predict temperatures during flow boiling experiments. The focus is on comparing heat transfer enhancements on smooth and needle-like microstructured surfaces.

# Project Overview
The objective of this project is to predict temperatures and analyze heat transfer behavior under various experimental conditions. Key features include:

* Data preprocessing and transformation
* Machine learning models (e.g., Random Forest, k-means)
* Analysis of boiling phenomena using voltage, temperature, and derived metrics
* Clustering and classification of flow phases

# Project Structure
**time_domain.py:**

* Visualizes temperature vs. time for smooth and microstructured surfaces during laminar and turbulent flow conditions.
* Helps identify patterns and variations in temperature over time.

**two_models.py:**

* Implements rolling statistics, lagged features, and clustering to compute boiling thresholds.
* Splits data into pre-boiling and boiling phases, trains separate Random Forest models, and evaluates them using metrics such as MSE, RMSE, and R².
* Generates summary metrics and plots for actual vs. predicted temperatures.

**utils.py:**

* Provides utility functions for preprocessing data, including extracting voltage information and handling missing values.
* Contains configuration constants like window size, random state, and test size.

**single_model_time_voltage.py:**

* Trains a single random forest model for each voltage level, evaluating its performance across multiple datasets.
* Generates summary metrics and plots for actual vs. predicted temperatures.

**single_model_V1_V2_T1.py:**

* Adds features for voltage and temperature differences (V1, V2, T1, T2) and trains a random forest model to predict T2 based on these inputs.
* Generates summary metrics and plots for actual vs. predicted temperatures.
