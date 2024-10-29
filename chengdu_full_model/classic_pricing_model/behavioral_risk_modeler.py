import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def run_classic_price_model():
    """
    Run a severity and frequency model on insurance data using Gamma and Poisson regressions.

    This function processes insurance data to predict total claim amounts (severity)
    and the frequency of claims. It performs the following tasks:
    - Loads and preprocesses data, including encoding categorical variables and
      handling missing values.
    - Splits data into training and test sets.
    - Trains a Gamma regression model for claim severity and evaluates model performance.
    - Trains a Poisson regression model for claim frequency and evaluates model performance.
    - Visualizes feature importance for both models based on model coefficients.

    Returns:
        None
    """

    # -----------------------------------
    # Data Loading and Preprocessing
    # -----------------------------------
    synthetic_data_path = 'data/synthetic_data.csv'

    # Load the data
    data = pd.read_csv(synthetic_data_path)

    # Selected features to include in the model
    selected_features = [
        'months_as_customer', 'age', 'policy_state', 'policy_deductable',
        'policy_annual_premium', 'policy_limit', 'umbrella_limit',
        'insured_sex', 'insured_education_level', 'insured_occupation',
        'insured_relationship', 'capital-gains', 'capital-loss',
        'incident_type', 'collision_type', 'incident_severity',
        'incident_hour_of_the_day', 'number_of_vehicles_involved',
        'bodily_injuries', 'witnesses', 'police_report_available',
        'auto_make', 'auto_model', 'auto_year', 'weather_condition',
        'previous_claims', 'avg_past_claim_amount', 'vehicle_age',
        'seasonal_factor', 'time_factor', 'weather_factor', 'tenure_days'
    ]

    # Drop rows with missing values for selected features and target variable
    data = data.dropna(subset=selected_features + ['total_claim_amount'])

    # Encode categorical variables using one-hot encoding
    data_encoded = pd.get_dummies(data[selected_features], drop_first=True)

    # Convert boolean columns to integer for compatibility
    bool_cols = data_encoded.select_dtypes(include=['bool']).columns
    data_encoded[bool_cols] = data_encoded[bool_cols].astype(int)

    # Target variable for severity
    target_severity = data['total_claim_amount']

    # -----------------------------------
    # Train Gamma Regression Model for Severity
    # -----------------------------------

    # Split the data into training and testing sets for severity
    X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(
        data_encoded, target_severity, test_size=0.2, random_state=42
    )

    # Add a constant term for the intercept in severity model
    X_train_sev_sm = sm.add_constant(X_train_sev)
    X_test_sev_sm = sm.add_constant(X_test_sev)

    # Convert all values to numeric and handle potential NaNs
    X_train_sev_sm = X_train_sev_sm.apply(
        pd.to_numeric, errors='coerce').fillna(0)
    X_test_sev_sm = X_test_sev_sm.apply(
        pd.to_numeric, errors='coerce').fillna(0)

    # Fit the Gamma regression model with a log link function for severity prediction
    gamma_model = sm.GLM(
        y_train_sev, X_train_sev_sm, family=sm.families.Gamma(
            link=sm.families.links.Log())
    )
    gamma_results = gamma_model.fit()

    # Display summary of the severity model
    print(gamma_results.summary())

    # Predict on the test set for severity
    y_pred_sev = gamma_results.predict(X_test_sev_sm)

    # Calculate and display performance metrics for severity
    mae_sev = mean_absolute_error(y_test_sev, y_pred_sev)
    rmse_sev = np.sqrt(mean_squared_error(y_test_sev, y_pred_sev))
    print(f"\nSeverity Model Performance Metrics:")
    print(f"Mean Absolute Error (MAE): {mae_sev:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse_sev:.2f}")

    # Calculate pseudo R^2 for model interpretability
    null_model_sev = sm.GLM(
        y_train_sev, sm.add_constant(np.ones_like(y_train_sev)),
        family=sm.families.Gamma(link=sm.families.links.Log())
    )
    null_results_sev = null_model_sev.fit()
    null_deviance_sev = null_results_sev.deviance
    model_deviance_sev = gamma_results.deviance
    pseudo_r2_sev = 1 - (model_deviance_sev / null_deviance_sev)
    print(f"Pseudo R^2: {pseudo_r2_sev:.4f}")

    # Feature importance based on model coefficients for severity
    coefficients_sev = gamma_results.params.drop('const')
    feature_importance_sev = coefficients_sev.abs().sort_values(ascending=False)

    # Display and visualize the top 10 features for severity
    print("\nTop 10 Most Relevant Features for Severity:")
    print(feature_importance_sev.head(10))
    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importance_sev.head(10),
                y=feature_importance_sev.head(10).index, palette='viridis')
    plt.title('Top 10 Most Relevant Features for Severity')
    plt.xlabel('Absolute Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('data/behavioral_important_features.png')
    plt.show()

    # -----------------------------------
    # Train Poisson Regression Model for Frequency
    # -----------------------------------

    # Define target variable for frequency
    target_freq = data['frequency']

    # Split the data into training and testing sets for frequency
    X_train_freq, X_test_freq, y_train_freq, y_test_freq = train_test_split(
        data_encoded, target_freq, test_size=0.2, random_state=42
    )

    # Add a constant term for the intercept in frequency model
    X_train_freq_sm = sm.add_constant(X_train_freq)
    X_test_freq_sm = sm.add_constant(X_test_freq)

    # Ensure numeric data and handle potential NaNs
    X_train_freq_sm = X_train_freq_sm.apply(
        pd.to_numeric, errors='coerce').fillna(0)
    X_test_freq_sm = X_test_freq_sm.apply(
        pd.to_numeric, errors='coerce').fillna(0)

    # Fit the Poisson regression model for frequency prediction
    poisson_model = sm.GLM(
        y_train_freq, X_train_freq_sm, family=sm.families.Poisson()
    )
    poisson_results = poisson_model.fit()

    # Display summary of the frequency model
    print(poisson_results.summary())

    # Predict on the test set for frequency
    y_pred_freq = poisson_results.predict(X_test_freq_sm)

    # Calculate and display performance metrics for frequency
    mae_freq = mean_absolute_error(y_test_freq, y_pred_freq)
    rmse_freq = np.sqrt(mean_squared_error(y_test_freq, y_pred_freq))
    print(f"\nFrequency Model Performance Metrics:")
    print(f"Mean Absolute Error (MAE): {mae_freq:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse_freq:.2f}")

    # Calculate pseudo R^2 for frequency model
    null_model_freq = sm.GLM(
        y_train_freq, sm.add_constant(np.ones_like(y_train_freq)),
        family=sm.families.Poisson()
    )
    null_results_freq = null_model_freq.fit()
    null_deviance_freq = null_results_freq.deviance
    model_deviance_freq = poisson_results.deviance
    pseudo_r2_freq = 1 - (model_deviance_freq / null_deviance_freq)
    print(f"Pseudo R^2: {pseudo_r2_freq:.4f}")

    # Feature importance based on model coefficients for frequency
    coefficients_freq = poisson_results.params.drop('const')
    feature_importance_freq = coefficients_freq.abs().sort_values(ascending=False)

    # Display and visualize the top 10 features for frequency
    print("\nTop 10 Most Relevant Features for Frequency:")
    print(feature_importance_freq.head(10))
    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importance_freq.head(10),
                y=feature_importance_freq.head(10).index, palette='magma')
    plt.title('Top 10 Most Relevant Features for Frequency')
    plt.xlabel('Absolute Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_frequency.png'))
    plt.show()
