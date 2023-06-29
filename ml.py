import pandas as pd
import datetime as dt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from pmdarima import auto_arima
import prophet
from scipy.stats.mstats import winsorize
import warnings

# Turn off all warnings
warnings.filterwarnings("ignore")

# date_column = "Date"
# regressor_columns = ['Covid', 'Price', 'other']
# products_column = "Sku"
# forecast_column = "Units"
# forecast_year = 2023
# file_content = "json_20230619144710.json"


def load_and_preprocess(df,products_column, product):
    df = df[df[products_column] == product]
    if df.empty:
        return df
    else:
        # Winsorize 'y' column by 10%
        winsorized_y = winsorize(df['y'], limits=(0.1, 0.1))
        df['y'] = winsorized_y
    return df.reset_index(drop=True)

def split_and_scale(df, forecast_year, forecast_month):
    # Sort df by 'ds' in ascending order
    df_sorted = df.sort_values('ds', ascending=True)
    df = df_sorted
    if isinstance(forecast_year, int):
        train_idx = df['DateOrdinal'] < dt.date(forecast_year, forecast_month, 1).toordinal()
        test_idx = df['DateOrdinal'] >= dt.date(forecast_year, forecast_month, 1).toordinal()
        return df, train_idx, test_idx
    else:
        try:
            if df['y'].isnull().any():
                # Find the index of the first empty value of 'y' from the bottom
                empty_y_index = df_sorted['y'].last_valid_index() + 1
                # Divide the data into train and test based on the empty_y_index
                train_idx = df_sorted.index < empty_y_index - 4
                test_idx = df_sorted.index >= empty_y_index - 4
                return df, train_idx, test_idx
            else:
                # Divide the data into train and test based on the last 4 rows
                
                train_idx = df.index < len(df) - 4
                test_idx = df.index >= len(df) - 4
                return df, train_idx, test_idx
        except Exception as e:
            return df, None, None
    if not any(train_idx) or not any(test_idx):
        return df, None, None
    return df.reset_index(drop=True), train_idx.reset_index(drop=True), test_idx.reset_index(drop=True)


def train_model(df, train_idx, regressor_columns, random_state, seasonality):
    train = df[train_idx]

    # Scale regressor columns
    scaler = StandardScaler()
    X_train_regressors_scaled = scaler.fit_transform(train[regressor_columns])
    X_train_regressors_scaled = pd.DataFrame(X_train_regressors_scaled, columns=regressor_columns)

    # Create feature sets
    X_train = pd.concat([X_train_regressors_scaled, train['Year']], axis = 1)
    X_train_ordinal = pd.concat([X_train_regressors_scaled, train['DateOrdinal']], axis=1)
    X_train_date =  train[['y', 'ds'] + regressor_columns]
    X_train_base = train[regressor_columns + ['Year'] + [f"Month_{i}" for i in range(1, 13)]]
    X_train_month = train[[f"Month_{i}" for i in range(1, 13)]]
    y_train = train['y']

    models = {
        'RandomForest': RandomForestRegressor(random_state=random_state),
        'XGBRegressor': XGBRegressor(random_state=random_state),
        'LinearRegression': LinearRegression(),
        'Auto_arima': auto_arima(y_train, exogenous=X_train_ordinal, seasonal=seasonality) if len(train) >= 15 else auto_arima(y_train, exogenous=X_train_ordinal, seasonal=False),
        'Prophet': prophet.Prophet(seasonality_mode='multiplicative') if len(train) >= 15 else prophet.Prophet()
    }
    trained_models = {}
    try:
        for model_name, model in models.items():
            if model_name == 'Auto_arima':
                model.fit(y_train, exogenous=X_train_ordinal)
            elif model_name == 'Prophet':
                for column in regressor_columns:
                    model = model.add_regressor(column)
                model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                model.fit(X_train_date)
            else:
                model.fit(X_train_base, y_train)
            trained_models[model_name] = model

        return trained_models
    except Exception as e:
        return trained_models

def predict_models(model, df, test_idx, regressor_columns):
    predictions = {}
    try:
        test = df[test_idx]

        # Scale regressor columns
        scaler = StandardScaler()
        X_test_regressors_scaled = scaler.fit_transform(test[regressor_columns])
        X_test_regressors_scaled = pd.DataFrame(X_test_regressors_scaled, columns=regressor_columns)

        # Create feature sets
        X_test = pd.concat([X_test_regressors_scaled, test['Year']], axis=1)
        X_test_ordinal = pd.concat([X_test_regressors_scaled, test['DateOrdinal']], axis=1)
        X_test_date = test[['ds'] + regressor_columns]
        X_test_base = test[regressor_columns + ['Year'] + [f"Month_{i}" for i in range(1, 13)]]
        X_test_month = test[[f"Month_{i}" for i in range(1, 13)]]
        print(X_test_base)
        predictions = {}
        try:
            for model_name, model in model.items():
                try:
                    if model_name == 'Auto_arima':
                        start_index = len(df) - len(test)
                        predictions[model_name] = model.predict(n_periods=len(test),
                                                            exogenous=test,
                                                            start=start_index)
                        predictions[model_name] = predictions[model_name].values
                    elif model_name == 'Prophet':
                        forecast = model.predict(X_test_date)
                        predictions[model_name] = forecast['yhat'].tail(len(X_test_date)).values
                    else:
                        predictions[model_name] = model.predict(X_test_base)
                    
                    # Calculate the mean value based on the original target variable
                    mean_prediction = np.mean(df['y'])
                    
                    # Assign 15% of the mean value to predictions less than 15% of the mean
                    predictions[model_name] = np.where(predictions[model_name] < 0.15 * mean_prediction, 0.15 * mean_prediction, predictions[model_name])
                except:
                    predictions[model_name] = None
            return predictions
        except Exception as e:
            return predictions
    except Exception as e:
        return predictions



def present_results(df, test_idx, predictions, user_prediction):
    try:
        results = df[test_idx].copy()
        results.reset_index(drop=True, inplace=True)

        # Evaluate MAE for last 5 data points by date for each model
        last_4_data = results.groupby('ds').head(4)
        for model_name, model_predictions in predictions.items():
            predicted_column_name = f'Predicted_{model_name}'
            predicted_values = model_predictions[-len(last_4_data):]
            results[predicted_column_name] = model_predictions
            accuracy = (1 - abs(results['y'] - model_predictions) / results['y']) * 100
            results[f'Accuracy_{model_name}'] = accuracy.round(1).clip(lower=0).astype(str) + "%"

        # Add the best model's prediction and accuracy
        best_model = min(predictions, key=lambda x: mean_absolute_error(results['y'], predictions[x]))
        results['BestModel'] = best_model
        results['Predicted_BestModel'] = predictions[best_model]
        best_model_accuracy = (1 - abs(results['y'] - predictions[best_model]) / results['y']) * 100
        results[f'Accuracy_BestModel'] = best_model_accuracy.round(1).clip(lower=0).astype(str) + "%"
        
        if user_prediction:
            user_prediction_values = results[user_prediction]
            if user_prediction_values.isna().all():
                results[f'{user_prediction} accuracy'] = None
            else:
                try:
                    user_forecast = (1 - abs(results['y'] - user_prediction_values) / results['y']) * 100
                    results[f'{user_prediction} accuracy'] = user_forecast.round(1).clip(lower=0).astype(str) + "%"
                except Exception as e:
                    pass
        return results
    except Exception as e:
        print("Exception:", e)
        return None

    
def Forecasted_Algorithms(file_content, date_column, regressor_columns, products_column, forecast_column,
                          user_prediction, forecast_year, forecast_month, products_to_forecast):
    selected_columns = [date_column] + regressor_columns + [products_column, forecast_column] + [user_prediction]
    print(selected_columns)
    print(forecast_year)
    print(forecast_month)
    df_core = pd.read_json(file_content, orient='records', convert_dates=[date_column])
    df = df_core[selected_columns]
    df[regressor_columns] = df[regressor_columns].fillna(0)
    columns_to_dropna = [date_column] + regressor_columns + [products_column, forecast_column]
    df[columns_to_dropna] = df[columns_to_dropna].dropna()

    df = df.rename(columns={date_column: 'ds', forecast_column: 'y'})
    df['DateOrdinal'] = df['ds'].map(dt.datetime.toordinal)
    df['Year'] = df['ds'].dt.year
    df['Month'] = df['ds'].dt.month
        # Add month columns
    for month in range(1, 13):
        df[f'Month_{month}'] = (df['Month'] == month).astype(int)
    products = df_core[products_to_forecast].unique()
    print(products)
    all_results = []
    for product in products:
        print(product)
        product_df = load_and_preprocess(df,products_column ,product)
        if not product_df.empty:
            preprocessed_df, train_idx, test_idx = split_and_scale(product_df, forecast_year=forecast_year, forecast_month=forecast_month)
            print(preprocessed_df[test_idx])
            if train_idx is not None and test_idx is not None:
                trained_models = train_model(preprocessed_df, train_idx, regressor_columns=regressor_columns,
                                             random_state=None , seasonality=12)
                print(trained_models)
                predictions = predict_models(trained_models, preprocessed_df, test_idx, regressor_columns)
                print(predictions)
                results = present_results(preprocessed_df, test_idx, predictions, user_prediction)
                if results is not None:
                    print(results,  len(results))
                    all_results.append(results)
                    all_results.append(preprocessed_df[train_idx][preprocessed_df[train_idx]['ds'].dt.year > 2022])
    try:
        if all_results:
            all_results_df = pd.concat(all_results, ignore_index=True)
            # Filter the DataFrame to include only rows after 2021
            #all_results_df = all_results_df[all_results_df['ds'].dt.year > 2021]
            json_data = all_results_df.to_json(date_format='iso', orient='records')
            return json_data
        else:
            return None
    except Exception as e:
        print("Exception:", e)
        return None  
