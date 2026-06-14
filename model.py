import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def build_features(monthly_df):
    df = monthly_df.copy()
    df['Month_Num'] = range(len(df))          # trend index
    df['Sin_Month'] = np.sin(2 * np.pi * df['Month'] / 12)   # seasonality
    df['Cos_Month'] = np.cos(2 * np.pi * df['Month'] / 12)
    return df


def train_model(df):
    features = ['Month_Num', 'Year', 'Month', 'Sin_Month', 'Cos_Month']
    X = df[features]
    y = df['Sales']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))

    return model, preds, mae, rmse


def forecast_future(model, df, months_ahead=12):
    last_row = df.iloc[-1]
    last_month_num = last_row['Month_Num']
    last_date = last_row['Date']

    future_records = []
    for i in range(1, months_ahead + 1):
        future_date = last_date + pd.DateOffset(months=i)
        month_num = last_month_num + i
        month = future_date.month
        year = future_date.year
        future_records.append({
            'Date': future_date,
            'Month_Num': month_num,
            'Year': year,
            'Month': month,
            'Sin_Month': np.sin(2 * np.pi * month / 12),
            'Cos_Month': np.cos(2 * np.pi * month / 12),
        })

    future_df = pd.DataFrame(future_records)
    features = ['Month_Num', 'Year', 'Month', 'Sin_Month', 'Cos_Month']
    future_df['Predicted_Sales'] = model.predict(future_df[features])

    return future_df
