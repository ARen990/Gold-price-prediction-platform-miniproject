import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import pickle
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from sklearn.feature_selection import SelectKBest, f_regression
from datetime import datetime
from pathlib import Path
import time
import os
import random

warnings.filterwarnings("ignore")
MODEL_DIR = Path(r"D:\project\artifacts\model\long")
if not MODEL_DIR.exists():
    for alt in [Path("/opt/artifacts/model/long"), Path("artifacts/model/long"), Path(".")]:
        if alt.exists():
            MODEL_DIR = alt
            break
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- ADD THIS CODE TO CONTROL RANDOMNESS ---
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# =========================================================================================================
# Data Downloading and Preprocessing Functions
# =========================================================================================================

def download_enhanced_data(symbols,end_date=None):
    print(">> Downloading 5 years of financial data with technical indicators...")
    frames = []

    for sym in symbols:
        df = yf.download(sym, period="5y", interval="1d", auto_adjust=False, progress=False)
        if df.empty:
            print(f"[WARN] no data for {sym}")
            continue
        
        # Basic OHLC data
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.columns = [f"{sym}_open", f"{sym}_high", f"{sym}_low", f"{sym}_close", f"{sym}_volume"]
        
        # Add technical indicators for each symbol
        df = add_technical_indicators(df, sym)
        frames.append(df)

    if not frames:
        raise SystemExit("[FATAL] No data downloaded.")

    df_raw = pd.concat(frames, axis=1).sort_index()
    df_num = df_raw.apply(pd.to_numeric, errors="coerce")
    threshold = 70
    mask_keep = df_num.isna().sum(axis=1) <= threshold
    df = df_num[mask_keep].copy()
    df = df.interpolate(method="linear", limit_direction="both").ffill().bfill()
    if end_date is not None:
        df = df[df.index <= pd.to_datetime(end_date)]

    print(f"Original rows: {len(df_raw)}, After cleaning: {len(df)}")
    
    # Add macroeconomic features
    df = add_macro_features(df)
    
    return df

def add_technical_indicators(df, symbol):
    """Add technical indicators for each symbol"""
    close_col = f"{symbol}_close"
    high_col = f"{symbol}_high"
    low_col = f"{symbol}_low"
    volume_col = f"{symbol}_volume"
    
    # Only add indicators if we have the required columns
    if close_col not in df.columns:
        return df
    
    # Moving averages
    df[f'{symbol}_sma_20'] = df[close_col].rolling(window=20).mean()
    df[f'{symbol}_ema_12'] = df[close_col].ewm(span=12).mean()
    df[f'{symbol}_ema_26'] = df[close_col].ewm(span=26).mean()
    
    # MACD
    df[f'{symbol}_macd'] = df[f'{symbol}_ema_12'] - df[f'{symbol}_ema_26']
    df[f'{symbol}_macd_signal'] = df[f'{symbol}_macd'].ewm(span=9).mean()
    df[f'{symbol}_macd_hist'] = df[f'{symbol}_macd'] - df[f'{symbol}_macd_signal']
    
    # RSI
    delta = df[close_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df[f'{symbol}_rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df[f'{symbol}_bb_middle'] = df[close_col].rolling(window=20).mean()
    bb_std = df[close_col].rolling(window=20).std()
    df[f'{symbol}_bb_upper'] = df[f'{symbol}_bb_middle'] + (bb_std * 2)
    df[f'{symbol}_bb_lower'] = df[f'{symbol}_bb_middle'] - (bb_std * 2)
    df[f'{symbol}_bb_width'] = df[f'{symbol}_bb_upper'] - df[f'{symbol}_bb_lower']
    
    # Additional indicators using high and low prices
    if all(col in df.columns for col in [high_col, low_col, close_col]):
        # Average True Range (ATR)
        tr1 = df[high_col] - df[low_col]
        tr2 = abs(df[high_col] - df[close_col].shift())
        tr3 = abs(df[low_col] - df[close_col].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df[f'{symbol}_atr'] = tr.rolling(window=14).mean()
        
        # Stochastic Oscillator
        low_14 = df[low_col].rolling(window=14).min()
        high_14 = df[high_col].rolling(window=14).max()
        df[f'{symbol}_stoch_k'] = 100 * ((df[close_col] - low_14) / (high_14 - low_14))
        df[f'{symbol}_stoch_d'] = df[f'{symbol}_stoch_k'].rolling(window=3).mean()
    

    # Volume indicators
    if volume_col in df.columns:
        df[f'{symbol}_volume_sma'] = df[volume_col].rolling(window=20).mean()
        df[f'{symbol}_volume_ratio'] = df[volume_col] / df[f'{symbol}_volume_sma']
    
    # Price rate of change
    df[f'{symbol}_roc'] = df[close_col].pct_change(periods=10)
    
    # Volatility
    df[f'{symbol}_volatility'] = df[close_col].pct_change().rolling(window=20).std()
    
    return df

def add_macro_features(df):
    """Add macroeconomic and time-based features"""
    # Time-based features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    
    # Rolling statistics for gold (as market indicator)
    if 'GC=F_close' in df.columns:
        df['gold_returns_30d'] = df['GC=F_close'].pct_change(30)
        df['gold_volatility_30d'] = df['GC=F_close'].pct_change().rolling(30).std()
    
    return df

def feature_selection(X_train, y_train, X_test, n_features=40):
    """Select top features using statistical tests"""
    print(f">> Selecting top {n_features} features...")
    
    # Handle NaN values
    X_train_filled = X_train.fillna(method='ffill').fillna(method='bfill').fillna(0)
    y_train_filled = y_train.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    selector = SelectKBest(score_func=f_regression, k=min(n_features, X_train_filled.shape[1]))
    X_train_selected = selector.fit_transform(X_train_filled, y_train_filled.mean(axis=1))
    X_test_selected = selector.transform(X_test.fillna(method='ffill').fillna(method='bfill').fillna(0))
    
    selected_features = X_train.columns[selector.get_support()]
    print(f"Selected {len(selected_features)} features")
    # print(selected_features.tolist())
    
    X_train_df = pd.DataFrame(X_train_selected, index=X_train.index, columns=selected_features)
    X_test_df = pd.DataFrame(X_test_selected, index=X_test.index, columns=selected_features)
    
    return X_train_df, X_test_df, selected_features

# =========================================================================================================
# Model Training Functions
# =========================================================================================================

def train_sarimax(X_train, y_train, X_test, targets, predictions, models):
    print("\n>> Training SARIMAX models on training data...")
    training_times = {}
    
    for target_col in targets:
        print(f"  -> Training & Forecasting SARIMAX for {target_col}")

        try:
            # Use a simpler, more robust approach
            best_order = (1, 0, 1)
            
            start_time = time.time()
            sarimax_model = SARIMAX(
                endog=y_train[target_col].values,
                exog=X_train.values if not X_train.empty else None,
                order=best_order
            )
            sarimax_result = sarimax_model.fit(disp=False, maxiter=50)
            training_time = time.time() - start_time
            
            models['SARIMAX'][target_col] = sarimax_result

            # Generate forecast
            start_time = time.time()
            if not X_test.empty:
                forecast = sarimax_result.forecast(
                    steps=len(X_test), 
                    exog=X_test.values if not X_test.empty else None
                )
            else:
                forecast = sarimax_result.forecast(steps=len(X_test))
            inference_time = time.time() - start_time
            
            # Create a proper pandas Series with correct index
            forecast_series = pd.Series(
                forecast, 
                index=X_test.index[:len(forecast)] if not X_test.empty else pd.RangeIndex(len(forecast))
            )
            
            predictions['SARIMAX'][target_col] = forecast_series
            training_times[f"SARIMAX_{target_col}"] = {'training': training_time, 'inference': inference_time}
            print(f"  [SUCCESS] SARIMAX forecast generated for {target_col} - Training: {training_time:.2f}s, Inference: {inference_time:.2f}s")
            
        except Exception as e:
            print(f"  [ERROR] SARIMAX failed for {target_col}: {e}")
            # Create a dummy forecast with NaN values
            predictions['SARIMAX'][target_col] = pd.Series(
                np.nan, 
                index=X_test.index
            )

    return models, predictions, training_times


def train_bvar(X_train, y_train, X_test, targets, predictions, models):
    print("\n>> Training a separate Bayesian VAR model for each target...")
    training_times = {}
    
    # Use more features for BVAR
    var_feature_cols = [col for col in X_train.columns if any(x in col for x in ['close', 'sma', 'rsi', 'macd'])]
    if len(var_feature_cols) > 20:  # Limit features for BVAR stability
        var_feature_cols = var_feature_cols[:20]
    
    X_train_var = X_train[var_feature_cols]
    X_test_var = X_test[var_feature_cols]

    for target_col in targets:
        print(f"  -> Training & Forecasting BVAR for {target_col}")
        var_df_train = pd.concat([y_train[[target_col]], X_train_var], axis=1)

        try:
            start_time = time.time()
            var_model = VAR(var_df_train)
            # Use smaller maxlags for stability
            var_result = var_model.fit(maxlags=5, ic='aic')
            training_time = time.time() - start_time
            
            models['Bayesian VAR'][target_col] = var_result

            lag_order = var_result.k_ar
            forecast_input = var_df_train.values[-lag_order:]
            
            start_time = time.time()
            var_forecast = var_result.forecast(y=forecast_input, steps=len(X_test_var))
            inference_time = time.time() - start_time

            var_forecast_df = pd.DataFrame(var_forecast, index=X_test.index, columns=var_df_train.columns)
            predictions['Bayesian VAR'][target_col] = var_forecast_df[target_col]
            training_times[f"BVAR_{target_col}"] = {'training': training_time, 'inference': inference_time}
            print(f"  [SUCCESS] BVAR forecast generated for {target_col} - Training: {training_time:.2f}s, Inference: {inference_time:.2f}s")

        except Exception as e:
            print(f"  [WARN] BVAR model failed to fit for {target_col}. Error: {e}")
            continue
    return models, predictions, training_times

def train_enhanced_lstm(X_train, y_train, X_test, targets, predictions, models):
    print("\n>> Training enhanced multi-output LSTM...")
    training_times = {}
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test)

    time_steps = 30  # Increased time steps

    def create_lstm_data(X, y, time_steps=time_steps):
        Xs, ys = [], []
        for i in range(time_steps, len(X)):
            Xs.append(X[i - time_steps:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    X_lstm_train, y_lstm_train = create_lstm_data(X_train_scaled, y_train_scaled, time_steps)
    num_targets = y_train.shape[1]

    # Enhanced LSTM architecture
    model = Sequential([
        LSTM(32, activation='relu', input_shape=(X_lstm_train.shape[1], X_lstm_train.shape[2])),
        Dropout(0.2),
        Dense(num_targets)
    ])
    # model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mse', metrics=['mae'])
    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Add validation split
    start_time = time.time()
    history = model.fit(
        X_lstm_train, y_lstm_train, 
        epochs=100, 
        batch_size=32, 
        verbose=0, 
        validation_split=0.2,
        callbacks=[es]
    )
    training_time = time.time() - start_time
    
    models['LSTM'] = model

    # Prepare test data for forecasting
    lstm_input = np.concatenate((X_train_scaled[-time_steps:], X_test_scaled))
    X_lstm_test, _ = create_lstm_data(lstm_input, np.zeros((len(lstm_input), num_targets)), time_steps)

    start_time = time.time()
    lstm_pred_scaled = model.predict(X_lstm_test)
    inference_time = time.time() - start_time
    
    lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled)
    lstm_pred_df = pd.DataFrame(lstm_pred, index=X_test.index, columns=targets)
    predictions['LSTM'] = lstm_pred_df
    training_times["LSTM"] = {'training': training_time, 'inference': inference_time}
    print(f"  [SUCCESS] LSTM forecast generated - Training: {training_time:.2f}s, Inference: {inference_time:.2f}s")

    scaler_y.data_max_ *= 1.5

    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('LSTM Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(MODEL_DIR / 'long_lstm_training_history.png')

    return models, predictions, scaler_X, scaler_y, history, training_times


def train_prophet(X_train, y_train, X_test, targets, predictions, models):
    print("\n>> Training Prophet models on training data...")
    training_times = {}
    
    for target_col in targets:
        print(f"  -> Preparing to train Prophet for {target_col}")
        prophet_train_df = pd.concat([y_train[[target_col]], X_train], axis=1)
        prophet_train_df = prophet_train_df.reset_index()
        prophet_train_df.rename(columns={prophet_train_df.columns[0]: 'ds', target_col: 'y'}, inplace=True)
        cleaned_df = prophet_train_df.dropna()

        if len(cleaned_df) < 2:
            print(f"  [WARN] Insufficient data for {target_col} after cleaning. Skipping Prophet model.")
            continue

        print(f"  -> Training & Forecasting Prophet for {target_col}")
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        prophet_features = [col for col in cleaned_df.columns if col not in ['ds', 'y']]
        for feature in prophet_features[:10]:  # Limit regressors for stability
            m.add_regressor(feature)

        start_time = time.time()
        m.fit(cleaned_df)
        training_time = time.time() - start_time
        
        models['Prophet'][target_col] = m

        future_df = X_test.copy().reset_index()
        future_df.rename(columns={future_df.columns[0]: 'ds'}, inplace=True)
        future_df.ffill(inplace=True)
        future_df.bfill(inplace=True)

        start_time = time.time()
        forecast = m.predict(future_df)
        inference_time = time.time() - start_time
        
        predictions['Prophet'][target_col] = forecast['yhat'].values
        training_times[f"Prophet_{target_col}"] = {'training': training_time, 'inference': inference_time}
        print(f"  [SUCCESS] Prophet forecast generated for {target_col} - Training: {training_time:.2f}s, Inference: {inference_time:.2f}s")

    return models, predictions, training_times


# =========================================================================================================
# Evaluation, Plotting, and Saving Functions
# =========================================================================================================
# def evaluate(y_true, y_pred, label, inference_time=None):
#     temp_df = pd.DataFrame({'true': y_true, 'pred': y_pred})
#     temp_df.dropna(inplace=True)
#     if temp_df.empty:
#         print(f"{label} -> No overlapping data to evaluate.")
#         return None, None, None
    
#     mae = mean_absolute_error(temp_df['true'], temp_df['pred'])
#     rmse = np.sqrt(mean_squared_error(temp_df['true'], temp_df['pred']) )
#     mape = mean_absolute_percentage_error(temp_df['true'], temp_df['pred'])
#     # print(f"{label} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2%}")

#     # Directional accuracy
#     direction_true = (temp_df['true'].pct_change() > 0).astype(int)
#     direction_pred = (temp_df['pred'].pct_change() > 0).astype(int)
#     direction_accuracy = (direction_true == direction_pred).mean()
#     # print(f"{label} -> Directional Accuracy: {direction_accuracy:.2%}")
    
#     correlation = temp_df['true'].corr(temp_df['pred'])

#     print(f"{label} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2%}, Dir_Acc: {direction_accuracy:.2%}")  
    
#     return mae, rmse, mape

def evaluate(y_true, y_pred, label, target_name):
    temp_df = pd.DataFrame({'true': y_true, 'pred': y_pred})
    temp_df.dropna(inplace=True)
    if temp_df.empty:
        print(f"{label} -> No overlapping data to evaluate.")
        return None, None, None, None
    
    mae = mean_absolute_error(temp_df['true'], temp_df['pred'])
    rmse = np.sqrt(mean_squared_error(temp_df['true'], temp_df['pred']))
    
    # Directional accuracy
    direction_true = (temp_df['true'].pct_change() > 0).astype(int)
    direction_pred = (temp_df['pred'].pct_change() > 0).astype(int)
    direction_accuracy = (direction_true == direction_pred).mean()
    
    mape = None
    smape = None

    # ADDED: Conditional metric calculation based on target type
    if 'spread' in target_name:
        numerator = 2 * np.abs(temp_df['pred'] - temp_df['true'])
        denominator = np.abs(temp_df['true']) + np.abs(temp_df['pred'])

        smape_vals = np.where(denominator == 0, 0, numerator / denominator)
        smape = np.mean(smape_vals) * 100
        print(f"{label} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, sMAPE: {smape:.2f}%, Dir_Acc: {direction_accuracy:.2%}")
    else:
        mape = mean_absolute_percentage_error(temp_df['true'], temp_df['pred'])
        print(f"{label} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2%}, Dir_Acc: {direction_accuracy:.2%}")
    
    return mae, rmse, mape, smape


def plot_results(y_train, y_test, predictions, targets, X_train):
    print("\n>> Generating enhanced forecast comparison plots...")
    for target_col in targets:
        plt.figure(figsize=(16, 8))
        # Reconstruct historical data for plotting if target is high or low
        if target_col == 'GC=F_high':
            y_train_plot = y_train['GC=F_close'] + y_train['GC=F_high_spread']
            y_test_plot = y_test['GC=F_close'] + y_test['GC=F_high_spread']
        elif target_col == 'GC=F_low':
            y_train_plot = y_train['GC=F_close'] - y_train['GC=F_low_spread']
            y_test_plot = y_test['GC=F_close'] - y_test['GC=F_low_spread']
        else: # This will be GC=F_close
            y_train_plot = y_train[target_col]
            y_test_plot = y_test[target_col]
            
        plt.plot(y_train.index, y_train_plot, label='Train Data', color='gray', alpha=0.7)
        plt.plot(y_test.index, y_test_plot, label='Actual (Test)', color='black', linewidth=2)

        colors = ['red', 'blue', 'green', 'orange']
        for i, (model_name, preds) in enumerate(predictions.items()):
            if 'GC=F_close' in preds.columns:
                # Reconstruct high/low from spreads for plotting
                if target_col == 'GC=F_high' and 'GC=F_high_spread' in preds.columns:
                    reconstructed_high = preds['GC=F_close'] + preds['GC=F_high_spread']
                    plt.plot(preds.index, reconstructed_high, label=f'{model_name} Forecast', 
                            linestyle='--', color=colors[i % len(colors)], alpha=0.8)
                elif target_col == 'GC=F_low' and 'GC=F_low_spread' in preds.columns:
                    reconstructed_low = preds['GC=F_close'] - preds['GC=F_low_spread']
                    plt.plot(preds.index, reconstructed_low, label=f'{model_name} Forecast', 
                            linestyle='--', color=colors[i % len(colors)], alpha=0.8)
                elif target_col == 'GC=F_close':
                    plt.plot(preds.index, preds[target_col], label=f'{model_name} Forecast', 
                            linestyle='--', color=colors[i % len(colors)], alpha=0.8)


        plt.axvline(X_train.index[-1], color='red', linestyle='--', label='Train/Test Split', linewidth=2)
        title_name = target_col.replace('GC=F_', '').replace('_', ' ').title()
        plt.title(f"Gold {title_name}: Enhanced Forecast vs. Actuals", fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price (USD)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def save_models(models, targets, scaler_X=None, scaler_y=None, selected_features=None):
    print("\n>> Saving all trained models and metadata...")
    
    # Convert selected_features to list if it's a pandas Index
    if selected_features is not None:
        selected_features_list = selected_features.tolist() if hasattr(selected_features, 'tolist') else list(selected_features)
    else:
        selected_features_list = []
    
    # Save model metadata
    metadata = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'targets': targets,
        'selected_features': selected_features_list
    }
    
    with open(MODEL_DIR / 'long_model_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    for target_col in targets:
        if target_col in models['SARIMAX']:
            with open(MODEL_DIR / f'long_sarimax_{target_col}.pkl', 'wb') as f:
                pickle.dump(models['SARIMAX'][target_col], f)
        if target_col in models['Prophet']:
            with open(MODEL_DIR / f'long_prophet_{target_col}.pkl', 'wb') as f:
                pickle.dump(models['Prophet'][target_col], f)
    
    for target_col in models['Bayesian VAR']:
        with open(MODEL_DIR / f'long_bvar_{target_col}.pkl', 'wb') as f:
            pickle.dump(models['Bayesian VAR'][target_col], f)
    
    if models['LSTM'] is not None:
        models['LSTM'].save(MODEL_DIR / 'long_lstm.keras')
    
    if scaler_X is not None and scaler_y is not None:
        # scaler_y.data_max_ *= 1.5
        with open(MODEL_DIR / 'long_scaler_X.pkl', 'wb') as f:
            pickle.dump(scaler_X, f)
        with open(MODEL_DIR / 'long_scaler_y.pkl', 'wb') as f:
            pickle.dump(scaler_y, f)
    
    print("\n--- All models and metadata saved successfully! ---")

# ========================================================================================================
# Walk-Forward Backtesting Function
# =========================================================================================================

def walk_forward_backtest(df, targets, base_features,
                        train_min_days=730,   
                        test_days=90,         
                        step_days=90,      
                        use_models=("SARIMAX", "Bayesian VAR", "LSTM", "Prophet")):

    all_metrics = []
    all_preds = {m: pd.DataFrame() for m in use_models}
    all_ytest = pd.DataFrame()

    dates = df.index.sort_values()
    start_idx = 0
    fold_count = 0

    while True:
        train_end_idx = min(len(dates)-1, start_idx + train_min_days - 1)
        test_start_idx = min(len(dates)-1, start_idx + train_min_days)
        if test_start_idx >= len(dates): 
            break

        test_end_idx = min(len(dates)-1, test_start_idx + test_days - 1)

        train_idx = dates[:test_start_idx]           # expanding window (>= train_min_days)
        test_idx  = dates[test_start_idx:test_end_idx+1]

        if len(test_idx) == 0:
            break

        X = df[base_features]
        y = df[targets]

        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test,  y_test  = X.loc[test_idx],  y.loc[test_idx]

        print(f"\n=== Fold {fold_count + 1} ===")
        print(f"Train: {train_idx[0].date()} to {train_idx[-1].date()} (n={len(train_idx)})")
        print(f"Test:  {test_idx[0].date()} to {test_idx[-1].date()} (n={len(test_idx)})")

        try:
            X_train_sel, X_test_sel, selected = feature_selection(X_train, y_train, X_test, n_features=50)
        except Exception as e:
            print(f"  [WARN] Feature selection failed: {e}. Using original features.")
            X_train_sel, X_test_sel = X_train, X_test
            selected = X_train.columns

        models = {"SARIMAX": {}, "Bayesian VAR": {}, "LSTM": None, "Prophet": {}}
        predictions = {
            "SARIMAX": pd.DataFrame(index=X_test_sel.index),
            "Bayesian VAR": pd.DataFrame(index=X_test_sel.index),
            "LSTM": pd.DataFrame(index=X_test_sel.index),
            "Prophet": pd.DataFrame(index=X_test_sel.index)
        }

        scalers = {}
        fold_training_times = {}
        
        if "SARIMAX" in use_models:
            try:
                models, predictions, sarimax_times = train_sarimax(X_train_sel, y_train, X_test_sel, targets, predictions, models)
                fold_training_times.update(sarimax_times)
                print(f"  [SUCCESS] SARIMAX trained for all targets")
            except Exception as e:
                print(f"  [ERROR] SARIMAX training failed: {e}")
        
        if "Bayesian VAR" in use_models:
            try:
                models, predictions, bvar_times = train_bvar(X_train_sel, y_train, X_test_sel, targets, predictions, models)
                fold_training_times.update(bvar_times)
                print(f"  [SUCCESS] Bayesian VAR trained for all targets")
            except Exception as e:
                print(f"  [ERROR] Bayesian VAR training failed: {e}")
        
        if "LSTM" in use_models:
            try:
                models, predictions, scaler_X, scaler_y, lstm_history, lstm_times = train_enhanced_lstm(
                    X_train_sel, y_train, X_test_sel, targets, predictions, models
                )
                scalers['LSTM'] = (scaler_X, scaler_y)
                fold_training_times.update(lstm_times)
                print(f"  [SUCCESS] LSTM trained for all targets")
            except Exception as e:
                print(f"  [ERROR] LSTM training failed: {e}")
        
        if "Prophet" in use_models:
            try:
                models, predictions, prophet_times = train_prophet(X_train_sel, y_train, X_test_sel, targets, predictions, models)
                fold_training_times.update(prophet_times)
                print(f"  [SUCCESS] Prophet trained for all targets")
            except Exception as e:
                print(f"  [ERROR] Prophet training failed: {e}")

        fold_metrics = {
            "fold": fold_count,
            "train_start": train_idx[0],
            "train_end": train_idx[-1], 
            "test_start": test_idx[0],
            "test_end": test_idx[-1],
            "train_days": len(train_idx),
            "test_days": len(test_idx)
        }
        
        # Add training times to metrics
        for time_key, time_val in fold_training_times.items():
            fold_metrics[f"{time_key}_training_time"] = time_val['training']
            fold_metrics[f"{time_key}_inference_time"] = time_val['inference']
        
        for model_name in use_models:
            if predictions[model_name].empty: 
                print(f"  [WARN] No predictions from {model_name}")
                continue
                
            for tgt in targets:
                if tgt not in predictions[model_name].columns: 
                    continue
                    
                y_true = y_test[tgt].reindex(predictions[model_name].index)
                y_pred = predictions[model_name][tgt]
                
                # Remove any remaining NaN values
                valid_mask = ~(y_true.isna() | y_pred.isna())
                y_true_clean = y_true[valid_mask]
                y_pred_clean = y_pred[valid_mask]
                
                if len(y_true_clean) == 0:
                    print(f"  [WARN] No valid data for {model_name}-{tgt}")
                    continue
                
                mae, rmse, mape = evaluate(y_true_clean, y_pred_clean, f"{model_name}-{tgt}")
                
                # Store metrics
                fold_metrics[f"{model_name}:{tgt}:MAE"] = mae
                fold_metrics[f"{model_name}:{tgt}:RMSE"] = rmse  
                fold_metrics[f"{model_name}:{tgt}:MAPE"] = mape
                
                # Store directional accuracy
                direction_true = (y_true_clean.pct_change() > 0).astype(int)
                direction_pred = (y_pred_clean.pct_change() > 0).astype(int)
                direction_accuracy = (direction_true == direction_pred).mean()
                fold_metrics[f"{model_name}:{tgt}:DIR_ACC"] = direction_accuracy

            all_preds[model_name] = pd.concat([all_preds[model_name], predictions[model_name]], axis=0)

        all_ytest = pd.concat([all_ytest, y_test.reindex(test_idx)], axis=0)
        all_metrics.append(fold_metrics)
        
        fold_count += 1

        start_idx = start_idx + step_days
        if start_idx + train_min_days >= len(dates): 
            break

    metrics_df = pd.DataFrame(all_metrics)
    
    # Calculate overall summary statistics and create visualizations
    summary_results = create_backtest_summary(metrics_df, all_preds, all_ytest, targets, use_models)
    
    return metrics_df, all_preds, all_ytest, summary_results


def create_backtest_summary(metrics_df, all_preds, all_ytest, targets, use_models):
    """Create comprehensive backtest summary with tables and plots"""
    
    print("\n" + "="*80)
    print("OVERALL BACKTESTING SUMMARY")
    print("="*80)
    
    summary_results = {}
    
    # Create summary tables
    summary_tables = {}
    
    for tgt in targets:
        print(f"\n📊 PERFORMANCE SUMMARY - {tgt}")
        print("-" * 60)
        
        table_data = []
        for model_name in use_models:
            # Get all metric columns for this model-target combination
            mae_cols = [col for col in metrics_df.columns if f"{model_name}:{tgt}:MAE" in col]
            rmse_cols = [col for col in metrics_df.columns if f"{model_name}:{tgt}:RMSE" in col]
            mape_cols = [col for col in metrics_df.columns if f"{model_name}:{tgt}:MAPE" in col]
            dir_acc_cols = [col for col in metrics_df.columns if f"{model_name}:{tgt}:DIR_ACC" in col]
            
            if mae_cols:
                mae_values = metrics_df[mae_cols[0]].dropna()
                rmse_values = metrics_df[rmse_cols[0]].dropna() if rmse_cols else pd.Series()
                mape_values = metrics_df[mape_cols[0]].dropna() if mape_cols else pd.Series()
                dir_acc_values = metrics_df[dir_acc_cols[0]].dropna() if dir_acc_cols else pd.Series()
                
                if len(mae_values) > 0:
                    model_stats = {
                        'Model': model_name,
                        'Folds': len(mae_values),
                        'MAE': f"{mae_values.mean():.4f} ± {mae_values.std():.4f}",
                        'RMSE': f"{rmse_values.mean():.4f} ± {rmse_values.std():.4f}" if len(rmse_values) > 0 else 'N/A',
                        'MAPE': f"{mape_values.mean():.2%} ± {mape_values.std():.2%}" if len(mape_values) > 0 else 'N/A',
                        'Dir_Acc': f"{dir_acc_values.mean():.2%}" if len(dir_acc_values) > 0 else 'N/A'
                    }
                    table_data.append(model_stats)
                    
                    summary_results[f"{model_name}_{tgt}"] = {
                        'MAE_mean': mae_values.mean(),
                        'MAE_std': mae_values.std(),
                        'RMSE_mean': rmse_values.mean() if len(rmse_values) > 0 else None,
                        'RMSE_std': rmse_values.std() if len(rmse_values) > 0 else None,
                        'MAPE_mean': mape_values.mean() if len(mape_values) > 0 else None,
                        'MAPE_std': mape_values.std() if len(mape_values) > 0 else None,
                        'DIR_ACC_mean': dir_acc_values.mean() if len(dir_acc_values) > 0 else None,
                        'n_folds': len(mae_values)
                    }
        
        # Display table
        if table_data:
            summary_df = pd.DataFrame(table_data)
            print(summary_df.to_string(index=False))
            summary_tables[tgt] = summary_df
    
    # Create comparison plots
    create_backtest_plots(all_preds, all_ytest, targets, use_models)
    
    # Create metrics trend plots
    create_metrics_trend_plots(metrics_df, targets, use_models)
    
    return summary_results


def create_backtest_plots(all_preds, all_ytest, targets, use_models):
    """Create backtest comparison plots"""
    
    print("\n📈 GENERATING BACKTEST PLOTS...")
    
    for tgt in targets:
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Actual vs Predicted (all models)
        plt.subplot(2, 1, 1)
        plt.plot(all_ytest.index, all_ytest[tgt], label='Actual', color='black', linewidth=2, alpha=0.8)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        for i, model_name in enumerate(use_models):
            if model_name in all_preds and not all_preds[model_name].empty and tgt in all_preds[model_name].columns:
                plt.plot(all_preds[model_name].index, all_preds[model_name][tgt], 
                        label=f'{model_name} Pred', linestyle='--', color=colors[i % len(colors)], alpha=0.7)
        
        plt.title(f'Backtest Results: {tgt} - Actual vs Predicted', fontsize=14, fontweight='bold')
        plt.ylabel('Price', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Prediction Errors (all models)
        plt.subplot(2, 1, 2)
        errors_data = []
        model_names = []
        
        for model_name in use_models:
            if model_name in all_preds and not all_preds[model_name].empty and tgt in all_preds[model_name].columns:
                # Align indices
                common_idx = all_ytest.index.intersection(all_preds[model_name].index)
                if len(common_idx) > 0:
                    y_true_aligned = all_ytest.loc[common_idx, tgt]
                    y_pred_aligned = all_preds[model_name].loc[common_idx, tgt]
                    errors = y_true_aligned - y_pred_aligned
                    errors_data.append(errors)
                    model_names.append(model_name)
        
        if errors_data:
            plt.boxplot(errors_data, labels=model_names)
            plt.title(f'Prediction Errors: {tgt}', fontsize=14, fontweight='bold')
            plt.ylabel('Error (Actual - Predicted)', fontsize=12)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def create_metrics_trend_plots(metrics_df, targets, use_models):
    """Create plots showing metric trends across folds"""
    
    print("\n📊 GENERATING METRICS TREND PLOTS...")
    
    for tgt in targets:
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Metrics Trend Across Folds: {tgt}', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['MAE', 'RMSE', 'MAPE', 'DIR_ACC']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx//2, idx%2]
            
            for model_name in use_models:
                metric_col = f"{model_name}:{tgt}:{metric}"
                if metric_col in metrics_df.columns:
                    values = metrics_df[metric_col].dropna()
                    if len(values) > 0:
                        ax.plot(range(len(values)), values, marker='o', label=model_name, alpha=0.7)
            
            ax.set_title(f'{metric} Trend')
            ax.set_xlabel('Fold Number')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# =========================================================================================================
# Main Execution
# =========================================================================================================

def main():
    symbols = ["GC=F", "BTC-USD", "^GSPC", "SLV", "EURUSD=X", "^DJI"]
    # Use END_DATE env var when provided, otherwise default to today's date (UTC)
    end_date_env = os.environ.get('END_DATE', '')
    if end_date_env:
        end_dt = pd.to_datetime(end_date_env)
        print(f">> Using END_DATE from environment: {end_dt.date()}")
    else:
        end_dt = pd.to_datetime(datetime.utcnow().date())
        print(f">> No END_DATE provided; defaulting to today's date (UTC): {end_dt.date()}")

    # Pass end date to the downloader so trimming happens early
    df = download_enhanced_data(symbols, end_date=end_dt)

    # ------------------- TARGET TRANSFORMATION -------------------
    print(">> Transforming targets to Close, High-Spread, and Low-Spread...")
    # Calculate spreads
    df['GC=F_high_spread'] = (df['GC=F_high'] - df['GC=F_close']).clip(lower=0)
    df['GC=F_low_spread'] = (df['GC=F_close'] - df['GC=F_low']).clip(lower=0)

    # Define the new targets
    targets = ['GC=F_close', 'GC=F_high_spread', 'GC=F_low_spread']
    
    # Define features: all columns EXCEPT the new targets AND the original high/low
    original_price_targets = ['GC=F_close', 'GC=F_high', 'GC=F_low']
    features_to_exclude = targets + [col for col in original_price_targets if col not in targets]
    features = [col for col in df.columns if col not in features_to_exclude]
    
    y = df[targets]
    X = df[features]

    # Set to True to enable walk-forward backtesting
    USE_WALK_FORWARD = False

    if not USE_WALK_FORWARD:
        split_idx = int(len(df) * 0.8)
        X_train_original, X_test_original = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        X_train, X_test, selected_features = feature_selection(
            X_train_original, y_train, X_test_original, n_features=40
        )

        print(f"\nData split into {len(X_train)} training samples and {len(X_test)} testing samples.")
        print(f"Using {len(selected_features)} selected features")

        models = {'SARIMAX': {}, 'Bayesian VAR': {}, 'LSTM': None, 'Prophet': {}}
        predictions = {m: pd.DataFrame(index=X_test.index) for m in ['SARIMAX', 'Bayesian VAR', 'LSTM', 'Prophet']}

        # Track training times
        all_training_times = {}
        
        models, predictions, sarimax_times = train_sarimax(X_train, y_train, X_test, targets, predictions, models)
        all_training_times.update(sarimax_times)
        
        models, predictions, bvar_times = train_bvar(X_train, y_train, X_test, targets, predictions, models)
        all_training_times.update(bvar_times)
        
        models, predictions, scaler_X, scaler_y, lstm_history, lstm_times = train_enhanced_lstm(X_train, y_train, X_test, targets, predictions, models)
        all_training_times.update(lstm_times)
        
        models, predictions, prophet_times = train_prophet(X_train, y_train, X_test, targets, predictions, models)
        all_training_times.update(prophet_times)
        
        # Print training times summary
        print("\n⏱ TRAINING AND INFERENCE TIMES:")
        print("-" * 50)
        for model_time, times in all_training_times.items():
            print(f"  {model_time:<20}: Training: {times['training']:.2f}s, Inference: {times['inference']:.2f}s")
        print("-" * 50)
        
        print("\n--- Enhanced Out-of-Sample Evaluation (Test Set) ---")
        evaluation_results = {}

        for target_col in targets:
            print(f"\n--- Evaluation for: {target_col} ---")
            evaluation_results[target_col] = {}
            
            is_spread_target = any(spread in target_col for spread in ['high_spread', 'low_spread'])
            
            if target_col in predictions['SARIMAX'].columns:
                mae, rmse, mape, smape = evaluate(y_test[target_col], predictions['SARIMAX'][target_col], "SARIMAX     ", target_col)
                evaluation_results[target_col]['SARIMAX'] = {
                    'MAE': mae, 'RMSE': rmse, 'sMAPE' if is_spread_target else 'MAPE': smape if is_spread_target else mape
                }
            if target_col in predictions['Bayesian VAR'].columns:
                mae, rmse, mape, smape = evaluate(y_test[target_col], predictions['Bayesian VAR'][target_col], "Bayesian VAR", target_col)
                evaluation_results[target_col]['Bayesian VAR'] = {
                    'MAE': mae, 'RMSE': rmse, 'sMAPE' if is_spread_target else 'MAPE': smape if is_spread_target else mape
                }
            if target_col in predictions['LSTM'].columns:
                mae, rmse, mape, smape = evaluate(y_test[target_col], predictions['LSTM'][target_col], "LSTM        ", target_col)
                evaluation_results[target_col]['LSTM'] = {
                    'MAE': mae, 'RMSE': rmse, 'sMAPE' if is_spread_target else 'MAPE': smape if is_spread_target else mape
                }
            if target_col in predictions['Prophet'].columns:
                mae, rmse, mape, smape = evaluate(y_test[target_col], predictions['Prophet'][target_col], "Prophet     ", target_col)
                evaluation_results[target_col]['Prophet'] = {
                    'MAE': mae, 'RMSE': rmse, 'sMAPE' if is_spread_target else 'MAPE': smape if is_spread_target else mape
                }
                
        # plot_results(y_train, y_test, predictions, targets, X_train)
        plot_results(y_train, y_test, predictions, ['GC=F_close', 'GC=F_high', 'GC=F_low'], X_train)
        save_models(models, targets, scaler_X, scaler_y, selected_features)

        results = {
            "mode": "single_split",
            "models": models,
            "predictions": predictions,
            "evaluation_results": evaluation_results,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "targets": targets,
            "selected_features": selected_features,
            "lstm_history": lstm_history,
            "feature_names": list(X.columns),
            "training_times": all_training_times,
        }

    else:
        metrics_df, preds_by_model, y_concat, summary_results = walk_forward_backtest(
            df=df,
            targets = targets,
            base_features = features,
            train_min_days = 730, 
            test_days = 90,       
            step_days = 90,      
            use_models = ("SARIMAX", "Bayesian VAR", "LSTM", "Prophet") 
        )

        results = {
            "mode": "walk_forward",
            "metrics_per_fold": metrics_df,
            "preds_by_model": preds_by_model, 
            "y_concat": y_concat,
            "summary_results": summary_results,
            "targets": targets,
            "feature_names": list(X.columns),
        }

    return results

if __name__ == "__main__":
    results = main()