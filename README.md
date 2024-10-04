https://www.kaggle.com/code/kanncaa1/feature-selection-and-data-visualization
https://www.kaggle.com/code/vissutagunawanlim/notebookf3c5fb3f6b/edit

# EDA
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_data(file_path):
    """
    Load the dataset from a file.
    """
    # You might need to adjust this based on the file format
    return pd.read_csv(file_path)

def basic_info(df):
    """
    Display basic information about the dataset.
    """
    print(df.info())
    print("\nDataset Shape:", df.shape)
    print("\nColumn Types:\n", df.dtypes)
    print("\nSummary Statistics:\n", df.describe())

def check_missing_values(df):
    """
    Analyze and visualize missing values.
    """
    missing = df.isnull().sum()
    missing_percent = 100 * missing / len(df)
    missing_table = pd.concat([missing, missing_percent], axis=1, keys=['Missing Values', 'Percentage'])
    print(missing_table)
    
    plt.figure(figsize=(10, 6))
    plt.bar(missing_table.index, missing_table['Percentage'])
    plt.title('Percentage of Missing Values by Feature')
    plt.xlabel('Features')
    plt.ylabel('Percentage of Missing Values')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_label_distribution(df, target_column):
    """
    Plot the distribution of the target variable.
    """
    plt.figure(figsize=(10, 6))
    if df[target_column].dtype in ['int64', 'float64']:
        sns.histplot(df[target_column], kde=True)
        plt.title(f'Distribution of {target_column}')
    else:
        df[target_column].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {target_column} (Categorical)')
    plt.xlabel(target_column)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def plot_seasonal_decomposition(df, date_column, target_column):
    """
    Plot seasonal decomposition for time series data.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    df[date_column] = pd.to_datetime(df[date_column])
    df.set_index(date_column, inplace=True)
    df.sort_index(inplace=True)
    
    result = seasonal_decompose(df[target_column], model='additive', period=12)  # Adjust period as needed
    result.plot()
    plt.tight_layout()
    plt.show()

def correlation_analysis(df):
    """
    Perform correlation analysis and plot a heatmap.
    """
    corr = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

def vif_analysis(df):
    """
    Calculate Variance Inflation Factor for numerical features.
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    vif_data = pd.DataFrame()
    vif_data["Feature"] = numeric_cols
    vif_data["VIF"] = [variance_inflation_factor(df[numeric_cols].values, i) for i in range(len(numeric_cols))]
    print(vif_data.sort_values("VIF", ascending=False))

def outlier_analysis(df):
    """
    Perform outlier analysis using boxplots, IQR, and Z-score methods.
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Boxplot
    plt.figure(figsize=(12, 6))
    df[numeric_cols].boxplot()
    plt.title('Boxplot for Numerical Features')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    # IQR method
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"IQR Outliers in {col}: {len(outliers)}")
    
    # Z-score method
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    z_score_outliers = (z_scores > 3).sum()
    print("\nZ-score Outliers (|z| > 3):")
    print(z_score_outliers)

def feature_distribution(df):
    """
    Plot distribution of numerical features.
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    n_cols = 3
    n_rows = (len(numeric_cols) - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
    
    for j in range(i+1, len(axes)):
        axes[j].remove()
    
    plt.tight_layout()
    plt.show()

def categorical_feature_analysis(df):
    """
    Analyze categorical features.
    """
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in cat_cols:
        print(f"\nUnique values in {col}: {df[col].nunique()}")
        print(df[col].value_counts(normalize=True))
        
        plt.figure(figsize=(10, 6))
        df[col].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def time_based_analysis(df, date_column, target_column):
    """
    Perform time-based analysis for time series data.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    df.set_index(date_column, inplace=True)
    df.sort_index(inplace=True)
    
    # Resampling to different time periods
    daily = df[target_column].resample('D').mean()
    weekly = df[target_column].resample('W').mean()
    monthly = df[target_column].resample('M').mean()
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
    
    daily.plot(ax=ax1)
    ax1.set_title('Daily Trend')
    
    weekly.plot(ax=ax2)
    ax2.set_title('Weekly Trend')
    
    monthly.plot(ax=ax3)
    ax3.set_title('Monthly Trend')
    
    plt.tight_layout()
    plt.show()

def run_eda(file_path, target_column, date_column=None):
    """
    Run all EDA functions.
    """
    df = load_data(file_path)
    basic_info(df)
    check_missing_values(df)
    plot_label_distribution(df, target_column)
    if date_column:
        plot_seasonal_decomposition(df, date_column, target_column)
        time_based_analysis(df, date_column, target_column)
    correlation_analysis(df)
    vif_analysis(df)
    outlier_analysis(df)
    feature_distribution(df)
    categorical_feature_analysis(df)

run_eda('your_dataset.csv', 'target_column', 'date_column')
```


# Feature Engineering
```python
from datetime import datetime, timedelta

def preprocess_datetime(df, datetime_column):
    """
    Main function to preprocess the datetime column.
    """
    df = convert_to_datetime(df, datetime_column)
    df = set_datetime_index(df, datetime_column)
    df = add_time_features(df)
    df = add_seasonal_features(df)
    df = handle_daylight_saving(df)
    df = handle_missing_timestamps(df)
    return df

def convert_to_datetime(df, datetime_column):
    """
    Convert the datetime column to pandas datetime format.
    """
    df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')
    print(f"Converted {datetime_column} to datetime. Range: {df[datetime_column].min()} to {df[datetime_column].max()}")
    return df

def set_datetime_index(df, datetime_column):
    """
    Set the datetime column as the index of the dataframe.
    """
    df = df.set_index(datetime_column)
    df = df.sort_index()
    print(f"Set {datetime_column} as index and sorted.")
    return df

def add_time_features(df):
    """
    Add various time-based features that might be relevant for renewable energy.
    """
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
    
    # Time of day categories
    df['time_of_day'] = pd.cut(df.index.hour, 
                               bins=[-1, 6, 12, 18, 23], 
                               labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    
    print("Added time-based features: hour, day, month, year, dayofweek, quarter, is_weekend, time_of_day")
    return df

def add_seasonal_features(df):
    """
    Add seasonal features that might affect renewable energy production.
    """
    # Seasons (Northern Hemisphere)
    df['season'] = pd.cut(df.index.month, 
                          bins=[-1, 2, 5, 8, 11, 12], 
                          labels=['Winter', 'Spring', 'Summer', 'Autumn', 'Winter'])
    
    # Solar declination angle (approximation)
    df['solar_declination'] = 23.45 * np.sin(np.radians((360/365) * (df.index.dayofyear - 81)))
    
    # Day length (in hours, approximation)
    latitude = 40  # Example latitude, adjust as needed
    df['day_length'] = 24 - (24/np.pi) * np.arccos(
        (np.sin(np.radians(-0.83)) + np.sin(np.radians(latitude)) * np.sin(np.radians(df['solar_declination']))) / 
        (np.cos(np.radians(latitude)) * np.cos(np.radians(df['solar_declination'])))
    )
    
    print("Added seasonal features: season, solar_declination, day_length")
    return df

def handle_daylight_saving(df):
    """
    Handle daylight saving time transitions.
    """
    df['is_dst'] = df.index.dst().astype(int)
    
    # Identify duplicate hours during "fall back"
    df['is_duplicate_hour'] = df.index.duplicated(keep=False)
    
    print("Added daylight saving features: is_dst, is_duplicate_hour")
    return df

def handle_missing_timestamps(df, freq='1H'):
    """
    Identify and handle missing timestamps.
    """
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    missing_timestamps = full_range.difference(df.index)
    
    if len(missing_timestamps) > 0:
        print(f"Found {len(missing_timestamps)} missing timestamps.")
        # Add missing timestamps with NaN values
        df = df.reindex(full_range)
        # Optionally, you can fill NaN values here
        # df = df.fillna(method='ffill')  # Forward fill
    else:
        print("No missing timestamps found.")
    
    return df

def analyze_timestamp_gaps(df):
    """
    Analyze gaps between timestamps.
    """
    time_diff = df.index.to_series().diff()
    
    print("Timestamp gap analysis:")
    print(time_diff.describe())
    
    # Identify large gaps
    large_gaps = time_diff[time_diff > pd.Timedelta(hours=1)]
    if not large_gaps.empty:
        print("\nLarge gaps found:")
        print(large_gaps)
    else:
        print("\nNo large gaps found.")
    
    return df

# Example usage:
# df = pd.read_csv('your_energy_data.csv')
# df = preprocess_datetime(df, 'timestamp_column')
# analyze_timestamp_gaps(df)

def add_lag_features(df, target_column, lags=[1, 24, 48, 168]):
    """
    Add lag features for the target variable.
    
    Args:
    df (pd.DataFrame): The input dataframe
    target_column (str): The name of the target column
    lags (list): List of lag values to create features for
    
    Returns:
    pd.DataFrame: Dataframe with added lag features
    """
    for lag in lags:
        df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
    
    print(f"Added lag features for lags: {lags}")
    return df

def add_diff_features(df, target_column, diffs=[1, 24, 48, 168]):
    """
    Add difference features for the target variable.
    
    Args:
    df (pd.DataFrame): The input dataframe
    target_column (str): The name of the target column
    diffs (list): List of difference periods to create features for
    
    Returns:
    pd.DataFrame: Dataframe with added difference features
    """
    for diff in diffs:
        df[f'{target_column}_diff_{diff}'] = df[target_column].diff(diff)
    
    print(f"Added difference features for periods: {diffs}")
    return df

def add_rolling_features(df, target_column, windows=[6, 12, 24, 48]):
    """
    Add rolling statistical features for the target variable.
    
    Args:
    df (pd.DataFrame): The input dataframe
    target_column (str): The name of the target column
    windows (list): List of rolling window sizes to create features for
    
    Returns:
    pd.DataFrame: Dataframe with added rolling features
    """
    for window in windows:
        df[f'{target_column}_rolling_mean_{window}'] = df[target_column].rolling(window=window).mean()
        df[f'{target_column}_rolling_std_{window}'] = df[target_column].rolling(window=window).std()
        df[f'{target_column}_rolling_min_{window}'] = df[target_column].rolling(window=window).min()
        df[f'{target_column}_rolling_max_{window}'] = df[target_column].rolling(window=window).max()
    
    print(f"Added rolling statistical features for windows: {windows}")
    return df

def add_expanding_features(df, target_column):
    """
    Add expanding (cumulative) statistical features for the target variable.
    
    Args:
    df (pd.DataFrame): The input dataframe
    target_column (str): The name of the target column
    
    Returns:
    pd.DataFrame: Dataframe with added expanding features
    """
    df[f'{target_column}_exp_mean'] = df[target_column].expanding().mean()
    df[f'{target_column}_exp_std'] = df[target_column].expanding().std()
    df[f'{target_column}_exp_min'] = df[target_column].expanding().min()
    df[f'{target_column}_exp_max'] = df[target_column].expanding().max()
    
    print("Added expanding statistical features")
    return df

def add_time_based_features(df):
    """
    Add more time-based features that might be relevant for renewable energy.
    
    Args:
    df (pd.DataFrame): The input dataframe
    
    Returns:
    pd.DataFrame: Dataframe with added time-based features
    """
    df['is_holiday'] = df.index.dayofweek.isin([5, 6]).astype(int)  # Simple weekend as holiday
    df['sin_hour'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['sin_day'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['cos_day'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    
    print("Added additional time-based features: is_holiday, sin_hour, cos_hour, sin_day, cos_day")
    return df

def preprocess_datetime_advanced(df, datetime_column, target_column):
    """
    Main function to preprocess the datetime column and add advanced time series features.
    """
    df = preprocess_datetime(df, datetime_column)
    df = add_time_based_features(df)
    df = add_lag_features(df, target_column)
    df = add_diff_features(df, target_column)
    df = add_rolling_features(df, target_column)
    df = add_expanding_features(df, target_column)
    return df

# Example usage:
# df = pd.read_csv('your_energy_data.csv')
# df = preprocess_datetime_advanced(df, 'timestamp_column', 'energy_output')
# analyze_timestamp_gaps(df)
```

# Modelling

```python 
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from catboost import CatBoostRegressor, Pool
import shap

def prepare_data(df, target_column, test_size=0.2, random_state=42):
    """
    Prepare the data for modeling by splitting into train and test sets.
    """
    # Assuming the index is already a datetime index
    train_df = df.iloc[:-int(len(df) * test_size)]
    test_df = df.iloc[-int(len(df) * test_size):]
    
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_evaluate_model_cv(merged_train, model, n_splits=5):
    X = merged_train.drop('% Baseline', axis=1)
    y = merged_train['% Baseline']
    
    kf = KFold(n_splits=n_splits, shuffle=False)
    
    rmse_scores = []
    feature_importances = []
    
    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_scores.append(rmse)
        
        feature_importances.append(get_feature_importance(model, X))
        
        print(f"Fold {fold} RMSE: {rmse}")
    
    avg_rmse = np.mean(rmse_scores)
    print(f"Average RMSE across {n_splits} folds: {avg_rmse}")
    
    avg_feature_importance = np.mean(feature_importances, axis=0)
    
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': avg_feature_importance
    })
    
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    # Define the color palette
    color_palette = ['#f7d01e', '#2b285c']

    plt.figure(figsize=(30, 25))
    
    # Create the bar plot with the new color
    sns.barplot(x='Importance', y='Feature', data=importance_df, color=color_palette[0])
    
    # Customize the plot
    plt.title('Average Feature Importance Across Folds', fontsize=24, color=color_palette[1])
    plt.xlabel('Importance', fontsize=18, color=color_palette[1])
    plt.ylabel('Features', fontsize=18, color=color_palette[1])
    
    # Adjust tick labels
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    # Add value labels to the end of each bar
    for i, v in enumerate(importance_df['Importance']):
        plt.text(v, i, f' {v:.3f}', va='center', fontsize=12, color=color_palette[1])
    
    # Set background color for contrast
    plt.gca().set_facecolor('#f0f0f0')
    
    # Add a grid for better readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    return avg_rmse, importance_df

def train_catboost_model(X_train, y_train, X_test, y_test, params=None):
    """
    Train a CatBoost model with cross-validation.
    """
    if params is None:
        params = {
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'random_seed': 42
        }
    
    # Time Series Cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    model = CatBoostRegressor(**params)
    
    # Train with cross-validation
    cv_results = model.select_features(
        X=X_train,
        y=y_train,
        eval_set=(X_test, y_test),
        features_for_select=X_train.columns.tolist(),
        num_features_to_select=20,
        steps=10,
        train_final_model=True,
        logging_level='Silent',
        plot=True
    )
    
    print("Cross-validation results:")
    print(cv_results)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    """
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    
    print(f"Test set RMSE: {rmse:.4f}")
    print(f"Test set MAE: {mae:.4f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, X):
    """
    Plot feature importance.
    """
    feature_importance = model.get_feature_importance()
    feature_names = X.columns
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20))
    plt.title("Top 20 Feature Importance")
    plt.tight_layout()
    plt.show()

def plot_shap_values(model, X):
    """
    Plot SHAP values.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X)
    plt.tight_layout()
    plt.show()

def run_modeling_pipeline(df, target_column):
    """
    Run the entire modeling pipeline.
    """
    # Prepare data
    X_train, X_val, y_train, y_val = prepare_data(df, target_column)
    
    # Train model
    model = train_catboost_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    evaluate_model(model, X_val, y_val)
    
    # Plot feature importance
    plot_feature_importance(model, X_train)
    
    # Plot SHAP values
    plot_shap_values(model, X_val)
    
    return model

# Example usage:
# df = pd.read_csv('your_preprocessed_energy_data.csv', parse_dates=['timestamp'], index_col='timestamp')
# model = run_modeling_pipeline(df, 'energy_output')
```
