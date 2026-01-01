import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import warnings
import matplotlib.pyplot as plt
from datetime import timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import traceback
import os
import time  # Added for timing

warnings.filterwarnings("ignore")

# --- PATHS (aligned with predict_long.py) ---
from pathlib import Path
import os

MODEL_DIR = Path(os.getenv("MODEL_DIR", r"D:\\project\\artifacts\\model\\long"))
OUT_DIR   = Path(os.getenv("OUT_DIR",   "artifacts/forecasts/long"))

if not MODEL_DIR.exists():
    for alt in [Path("/opt/artifacts/model/long"),
                Path("/artifacts/model/long"),
                Path("artifacts/model/long")]:
        if alt.exists():
            MODEL_DIR = alt
            break
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"📂 MODEL_DIR = {MODEL_DIR.resolve()}")
print(f"📂 OUT_DIR   = {OUT_DIR.resolve()}")



# ========================================================================================================
# Data Downloading and Preprocessing Functions
# ========================================================================================================
n_days = 180  # Forecast horizon
symbols = ["GC=F", "BTC-USD", "^GSPC", "SLV", "EURUSD=X", "^DJI"]
targets = ["GC=F_close", "GC=F_high_spread", "GC=F_low_spread"]
time_steps = 15

# ---------------------- Download Data ----------------------
print(">> Downloading latest market data...")
frames = []
for sym in symbols:
    df_sym = yf.download(sym, period="2y", interval="1d", auto_adjust=False, progress=False)
    if df_sym.empty:
        print(f"[WARN] No data for {sym}")
        continue
    df_sym = df_sym[["Open", "High", "Low", "Close", "Volume"]].copy()
    df_sym.columns = [f"{sym}_open", f"{sym}_high", f"{sym}_low", f"{sym}_close", f"{sym}_volume"]
    frames.append(df_sym)

if not frames:
    raise SystemExit("[FATAL] No data downloaded.")

df = pd.concat(frames, axis=1).sort_index()
df = df.apply(pd.to_numeric, errors="coerce")
df = df[df.isna().sum(axis=1) <= 20]
df = df.interpolate(method="linear", limit_direction="both").ffill().bfill()
print(f"✓ Cleaned data with {len(df)} rows.")

# ---------------------- Save Historical Data (only GC=F OHLC)----------------------
hist_outpath = OUT_DIR / "long_historical_data.csv"
df[["GC=F_open", "GC=F_high", "GC=F_low", "GC=F_close"]].to_csv(hist_outpath)
print(f"✓ Saved historical data to {hist_outpath}")

# ---------------------- Add Technical Indicators ----------------------
print(">> Adding technical indicators...")

def add_technical_indicators(df, symbol):
    close_col = f"{symbol}_close"
    high_col = f"{symbol}_high"
    low_col = f"{symbol}_low"
    volume_col = f"{symbol}_volume"
    
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
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    
    if 'GC=F_close' in df.columns:
        df['gold_returns_30d'] = df['GC=F_close'].pct_change(30)
        df['gold_volatility_30d'] = df['GC=F_close'].pct_change().rolling(30).std()
    
    return df

for symbol in symbols:
    df = add_technical_indicators(df, symbol)
df = add_macro_features(df)

# ---------------------- Calculate Spread Targets ----------------------
print(">> Calculating spread columns for model compatibility...")
if 'GC=F_high' in df.columns and 'GC=F_close' in df.columns:
    df['GC=F_high_spread'] = (df['GC=F_high'] - df['GC=F_close']).clip(lower=0)
if 'GC=F_close' in df.columns and 'GC=F_low' in df.columns:
    df['GC=F_low_spread'] = (df['GC=F_close'] - df['GC=F_low']).clip(lower=0)
print("✓ Spread columns calculated.")

# ---------------------- Load Model Metadata ----------------------
print(">> Loading model metadata...")
try:
    with open(MODEL_DIR / "long_model_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    selected_features = metadata.get('selected_features', [])
    print(f"✓ Loaded metadata with {len(selected_features)} selected features")
except FileNotFoundError:
    print("⚠️  Metadata not found, using all features")
    selected_features = [col for col in df.columns if col not in targets]


# ---------------------- Setup Features ----------------------
features = [c for c in df.columns if c not in targets]
if selected_features:
    features = [f for f in selected_features if f in df.columns]
    
X_hist = df[features].copy()


# ---------------------- Forecast Index ----------------------
forecast_index = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=n_days, freq='D')


# ---------------------- Random Walk for Exogenous Variables ----------------------
print(">> Generating future exogenous variables using Random Walk...")

def forecast_with_random_walk(series, steps, volatility_adjustment=1.0):
    """Forecast a series using random walk with drift"""
    if len(series) < 2:
        return np.full(steps, series.iloc[-1] if len(series) > 0 else 0)
    
    returns = series.pct_change().dropna()
    if len(returns) == 0:
        return np.full(steps, series.iloc[-1])
    
    # Calculate drift and volatility
    drift = returns.mean()
    volatility = returns.std() * volatility_adjustment
    
    # Generate forecast
    forecast = [series.iloc[-1]]
    for _ in range(1, steps):
        shock = np.random.normal(drift, volatility)
        next_val = forecast[-1] * (1 + shock)
        forecast.append(next_val)
    
    return np.array(forecast)

def create_future_features_random_walk(X_hist, features_list, periods=n_days):
    """Create future features using random walk for price-based variables"""
    future_dates = pd.date_range(start=X_hist.index[-1] + timedelta(days=1), periods=periods, freq='D')
    future_df = pd.DataFrame(index=future_dates)
    
    # Time-based features (known exactly)
    future_df['day_of_week'] = future_df.index.dayofweek
    future_df['month'] = future_df.index.month
    future_df['quarter'] = future_df.index.quarter
    future_df['year'] = future_df.index.year
    
    # Categorize features and handle them appropriately
    for feature in features_list:
        if feature not in future_df.columns:
            if feature in X_hist.columns:
                series = X_hist[feature].dropna()
                
                if len(series) > 10:  # Only forecast if we have enough data
                    if any(x in feature for x in ['_close', '_open', '_high', '_low']):
                        # Price data - use random walk
                        future_df[feature] = forecast_with_random_walk(series, periods, 0.8)
                    
                    elif any(x in feature for x in ['sma', 'ema', 'bb_middle']):
                        # Moving averages - smoother random walk
                        future_df[feature] = forecast_with_random_walk(series, periods, 0.3)
                    
                    elif 'rsi' in feature:
                        # RSI - mean revert around 50
                        last_rsi = series.iloc[-1]
                        # Mean reversion towards 50
                        reversion_speed = 0.1
                        rsi_forecast = []
                        current = last_rsi
                        for _ in range(periods):
                            current = current + reversion_speed * (50 - current) + np.random.normal(0, 2)
                            current = np.clip(current, 20, 80)  # Keep in reasonable range
                            rsi_forecast.append(current)
                        future_df[feature] = rsi_forecast
                    
                    elif any(x in feature for x in ['macd', 'volume_ratio', 'roc']):
                        # Oscillators and ratios - mean revert to 0 or 1
                        last_val = series.iloc[-1]
                        if 'volume_ratio' in feature:
                            target = 1.0
                        else:
                            target = 0.0
                        
                        reversion_forecast = []
                        current = last_val
                        for _ in range(periods):
                            current = current + 0.1 * (target - current) + np.random.normal(0, abs(current * 0.1))
                            reversion_forecast.append(current)
                        future_df[feature] = reversion_forecast
                    
                    elif any(x in feature for x in ['volatility', 'returns']):
                        # Volatility and returns - use recent average
                        recent_mean = series.tail(30).mean()
                        future_df[feature] = np.random.normal(recent_mean, abs(recent_mean * 0.2), periods)
                    
                    else:
                        # Default - use last value with small noise
                        last_val = series.iloc[-1]
                        future_df[feature] = last_val + np.random.normal(0, abs(last_val * 0.01), periods)
                else:
                    # Not enough data - use last value
                    future_df[feature] = series.iloc[-1] if len(series) > 0 else 0
    
    return future_df

# Generate future features using random walk
X_future = create_future_features_random_walk(X_hist, features, n_days)
X_future = X_future.reindex(columns=features).ffill().bfill().fillna(0)

# ---------------------- Predictions containers ----------------------
predictions = {
    "SARIMAX": pd.DataFrame(index=forecast_index, columns=targets),
    "Bayesian_VAR": pd.DataFrame(index=forecast_index, columns=targets),
    "Prophet": pd.DataFrame(index=forecast_index, columns=targets),
    "LSTM": pd.DataFrame(index=forecast_index, columns=targets),
}

# ---------------------- Inference Timing Measurements ----------------------
inference_times = {}


# ========================================================================================================
# Models Forecasting Section
# ========================================================================================================
# ---------------------- 1) SARIMAX ----------------------
print("🔮 Forecasting with SARIMAX...")
for target in targets:
    try:
        model_path = str(MODEL_DIR / f"long_sarimax_{target}.pkl")
        if not os.path.exists(model_path):
            print(f"  ⚠️  SARIMAX model not found for {target}")
            continue
            
        with open(model_path, "rb") as f:
            sarimax_res = pickle.load(f)
        
        start_time = time.time()
        fc = sarimax_res.get_forecast(steps=n_days, exog=X_future)
        inference_time = time.time() - start_time
        
        predictions["SARIMAX"][target] = fc.predicted_mean
        
        current_price = df[target].iloc[-1] if target in df.columns else np.nan
        forecast_price = predictions["SARIMAX"][target].iloc[-1]
        change_pct = ((forecast_price - current_price) / current_price * 100) if not np.isnan(current_price) else np.nan
        
        print(f"  ✅ SARIMAX {target}: ${forecast_price:.2f} ({change_pct:+.1f}%) - Time: {inference_time:.2f}s")
        inference_times[f"SARIMAX_{target}"] = inference_time
        
    except Exception as e:
        print(f"  ❌ SARIMAX failed for {target}: {e}")

# ---------------------- 2) Bayesian VAR ----------------------
print("🔮 Forecasting with Bayesian VAR...")
for target in targets:
    try:
        model_path = str(MODEL_DIR / f"long_bvar_{target}.pkl")
        if not os.path.exists(model_path):
            print(f"  ⚠️  BVAR model not found for {target}")
            continue
            
        with open(model_path, "rb") as f:
            bvar_res = pickle.load(f)
        
        # BVAR typically uses multiple time series
        # Get relevant features for this target
        var_feature_cols = [col for col in X_hist.columns if any(x in col for x in ['close', 'sma', 'rsi', 'macd'])]
        if len(var_feature_cols) > 15:
            var_feature_cols = var_feature_cols[:15]
        
        # Create the dataset BVAR was trained on
        bvar_data = pd.concat([df[[target]], X_hist[var_feature_cols]], axis=1).dropna()
        
        if len(bvar_data) > 0:
            # Get lag order
            k_ar = getattr(bvar_res, 'k_ar', 5)
            
            if len(bvar_data) >= k_ar:
                start_time = time.time()
                input_data = bvar_data.values[-k_ar:]
                forecast_result = bvar_res.forecast(input_data, steps=n_days)
                inference_time = time.time() - start_time
                
                if forecast_result.shape[1] == len(bvar_data.columns):
                    # The first column should be our target
                    predictions["Bayesian_VAR"][target] = forecast_result[:, 0]
                    
                    current_price = df[target].iloc[-1] if target in df.columns else np.nan
                    forecast_price = predictions["Bayesian_VAR"][target].iloc[-1]
                    change_pct = ((forecast_price - current_price) / current_price * 100) if not np.isnan(current_price) else np.nan
                    
                    print(f"  ✅ BVAR {target}: ${forecast_price:.2f} ({change_pct:+.1f}%) - Time: {inference_time:.2f}s")
                    inference_times[f"BVAR_{target}"] = inference_time
                else:
                    print(f"  ❌ BVAR {target}: Forecast shape mismatch")
            else:
                print(f"  ❌ BVAR {target}: Not enough data for lag order {k_ar}")
        else:
            print(f"  ❌ BVAR {target}: No training data available")
            
    except Exception as e:
        print(f"  ❌ BVAR failed for {target}: {e}")

# ---------------------- 3) Prophet ----------------------
print("🔮 Forecasting with Prophet...")
for target in targets:
    try:
        model_path = str(MODEL_DIR / f"long_prophet_{target}.pkl")
        if not os.path.exists(model_path):
            print(f"  ⚠️  Prophet model not found for {target}")
            continue
            
        with open(model_path, "rb") as f:
            pmodel = pickle.load(f)

        # Prepare future dataframe
        future_df = X_future.reset_index()
        future_df.rename(columns={'index': 'ds'}, inplace=True)
        
        # Prophet prediction
        start_time = time.time()
        forecast = pmodel.predict(future_df)
        inference_time = time.time() - start_time
        
        predictions["Prophet"][target] = forecast["yhat"].values
        
        current_price = df[target].iloc[-1] if target in df.columns else np.nan
        forecast_price = predictions["Prophet"][target].iloc[-1]
        change_pct = ((forecast_price - current_price) / current_price * 100) if not np.isnan(current_price) else np.nan
        
        print(f"  ✅ Prophet {target}: ${forecast_price:.2f} ({change_pct:+.1f}%) - Time: {inference_time:.2f}s")
        inference_times[f"Prophet_{target}"] = inference_time
        
    except Exception as e:
        print(f"  ❌ Prophet failed for {target}: {e}")

# ---------------------- 4) LSTM ----------------------
print("🔮 Forecasting with LSTM...")
try:
    # Try to load LSTM model
    lstm_model = None
    scaler_X = None
    scaler_y = None
    
    model_files = ["long_lstm.keras", "long_lstm.h5"]
    
    for model_file in model_files:
        full_path = MODEL_DIR / model_file
        if full_path.exists():
            try:
                lstm_model = load_model(str(full_path), compile=False)
                break
            except Exception as e:
                print(f"  ⚠️  Failed to load {model_file}: {e}")
                continue
    
    if lstm_model is None:
        print("  ⚠️  No LSTM model found")
    else:
        try:
            with open(MODEL_DIR / "long_scaler_X.pkl", "rb") as f:
                scaler_X = pickle.load(f)
            with open(MODEL_DIR / "long_scaler_y.pkl", "rb") as f:
                scaler_y = pickle.load(f)
            print("  ✓ Loaded scalers")
            
        except FileNotFoundError as e:
            print(f"  ❌ Scaler not found: {e}")
            lstm_model = None

    if lstm_model and scaler_X and scaler_y:
        # Prepare data - ensure no NaN values
        X_hist_clean = X_hist.fillna(method='ffill').fillna(method='bfill').fillna(0)
        X_future_clean = X_future.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Scale the data
        X_hist_scaled = scaler_X.transform(X_hist_clean)
        X_future_scaled = scaler_X.transform(X_future_clean)
        
        # Build input sequence and predict
        lstm_preds_scaled = []
        current_sequence = X_hist_scaled[-time_steps:].copy()
        
        start_time = time.time()
        for i in range(n_days):
            x_input = current_sequence.reshape(1, time_steps, len(features))
            y_pred_scaled = lstm_model.predict(x_input, verbose=0)
            
            # Handle model output
            if isinstance(y_pred_scaled, (list, tuple)):
                y_pred_scaled = y_pred_scaled[0]
            
            y_pred_flat = y_pred_scaled.flatten()
            n_targets_expected = min(len(targets), len(y_pred_flat))
            y_pred_final = y_pred_flat[:n_targets_expected]
            
            lstm_preds_scaled.append(y_pred_final)
            
            # Update sequence with future exogenous data
            if i < len(X_future_scaled):
                next_row = X_future_scaled[i].copy()
                current_sequence = np.vstack([current_sequence[1:], next_row])
        
        inference_time = time.time() - start_time
        print(f"  ✅ LSTM inference time: {inference_time:.2f}s")
        inference_times["LSTM"] = inference_time

        # Inverse transform to get actual price predictions
        lstm_preds = scaler_y.inverse_transform(lstm_preds_scaled)
        
                
        # Assign predictions directly
        for i, target in enumerate(targets):
            if i < lstm_preds.shape[1]:
                predictions["LSTM"][target] = lstm_preds[:, i]
                
                current_price = df[target].iloc[-1] if target in df.columns else np.nan
                forecast_price = predictions["LSTM"][target].iloc[-1] if len(predictions["LSTM"][target]) > 0 else np.nan
                
                if not np.isnan(forecast_price) and not np.isnan(current_price):
                    change_pct = ((forecast_price - current_price) / current_price * 100)
                    print(f"  ✅ LSTM {target}: ${forecast_price:.2f} ({change_pct:+.1f}%)")

except Exception as e:
    print(f"  ❌ LSTM prediction failed: {e}")
    traceback.print_exc()

# ========================================================================================================
# Reconstruct High/Low Prices from Spreads
# ========================================================================================================
print("\n🔧 Reconstructing High/Low prices from predicted spreads...")

for model_name, df_model in predictions.items():
    if df_model.empty or 'GC=F_close' not in df_model.columns:
        continue

    # Ensure spreads are non-negative
    if 'GC=F_high_spread' in df_model.columns:
        df_model['GC=F_high_spread'] = df_model['GC=F_high_spread'].clip(lower=0)
    if 'GC=F_low_spread' in df_model.columns:
        df_model['GC=F_low_spread'] = df_model['GC=F_low_spread'].clip(lower=0)
    
    # Reconstruct high and low prices
    if 'GC=F_high_spread' in df_model.columns and 'GC=F_low_spread' in df_model.columns:
        df_model['GC=F_high'] = df_model['GC=F_close'] + df_model['GC=F_high_spread']
        df_model['GC=F_low'] = df_model['GC=F_close'] - df_model['GC=F_low_spread']
        
        # Drop the spread columns as they are no longer needed
        df_model.drop(columns=['GC=F_high_spread', 'GC=F_low_spread'], inplace=True, errors='ignore')
        print(f"  ✓ Reconstructed prices for {model_name}")

# IMPORTANT: Update targets list for plotting and summary sections to use original price columns
targets = ["GC=F_close", "GC=F_high", "GC=F_low"]

# =========================================================================================================
# Evaluation, Plotting, and Saving Functions
# =========================================================================================================

print("\n💾 Saving forecasts to CSV...")
for model_name, df_model in predictions.items():
    if not df_model.empty and df_model.notna().any().any():
        filename = f"long_forecast_{model_name.lower().replace(' ', '_')}.csv"
        outpath = OUT_DIR / filename
        df_model.to_csv(outpath)
        print(f"  ✓ Saved {filename}")


print("📈 Plotting forecasts...")
for target in targets:
    if target not in df.columns:
        continue
        
    plt.figure(figsize=(16, 10))
    
    # Plot historical data
    hist_days = min(n_days, len(df))
    plt.plot(df.index[-hist_days:], df[target].iloc[-hist_days:], 
            label="Historical Data", color="black", linewidth=3, alpha=0.9)
    
    # Plot forecasts
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    linestyles = ['-', '--', '-', ':', '-']
    
    valid_models = 0
    for i, (model_name, df_model) in enumerate(predictions.items()):
        if target in df_model.columns and df_model[target].notna().any():
            plt.plot(df_model.index, df_model[target], 
                    label=f"{model_name}", 
                    color=colors[i % len(colors)], 
                    linestyle=linestyles[i % len(linestyles)],
                    linewidth=2.5,
                    alpha=0.8)
            valid_models += 1
    
    if valid_models > 0:
        plt.axvline(df.index[-1], color="red", linestyle="--", label="Forecast Start", linewidth=2, alpha=0.8)
        current_price = df[target].iloc[-1]
        plt.title(f"Gold {target.replace('GC=F_', '').replace('_', ' ').title()} - {n_days}-Day Forecast\nCurrent: ${current_price:.2f}", 
                    fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price (USD)", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"long_forecast_plot_{target}.png"
        plt.savefig(OUT_DIR / plot_filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved {OUT_DIR / plot_filename}")
        plt.show()
    else:
        print(f"  ⚠️  No valid forecasts for {target}, skipping plot")


# ========================================================================================================
# Final Summary
# ========================================================================================================
print("\n" + "="*80)
print(f"GOLD PRICE FORECAST SUMMARY - {n_days} DAYS")
print("="*80)

# Print inference times summary
print("\n>> INFERENCE TIMES:")
print("-" * 40)
for model_time, duration in inference_times.items():
    print(f"  {model_time:<20}: {duration:.2f}s")

print("\n" + "-" * 40)

for target in targets:
    if target in df.columns:
        current_price = df[target].iloc[-1]
        print(f"\n🎯 {target.replace('GC=F_', '').upper()}:")
        print(f"   Current: ${current_price:.2f}")
        print(f"   {'Model':<15} {f'{n_days}-Day':<12} {'Change %':<10} {'Trend'}")
        print(f"   {'-'*15} {'-'*12} {'-'*10} {'-'*10}")
        
        model_results = []
        for model_name in predictions.keys():
            if target in predictions[model_name].columns:
                forecast = predictions[model_name][target].iloc[-1]
                if not np.isnan(forecast):
                    change_pct = ((forecast - current_price) / current_price) * 100
                    trend = "↑ BULLISH" if change_pct > 5 else "↓ BEARISH" if change_pct < -5 else "→ NEUTRAL"
                    model_results.append((model_name, forecast, change_pct, trend))
        
        # Sort by forecast price
        model_results.sort(key=lambda x: x[1])
        
        for model_name, forecast, change_pct, trend in model_results:
            print(f"   {model_name:<15} ${forecast:<11.2f} {change_pct:<9.1f}% {trend}")
        
        if model_results:
            avg_forecast = np.mean([r[1] for r in model_results])
            avg_change = ((avg_forecast - current_price) / current_price) * 100
            print(f"\n   📊 CONSENSUS: ${avg_forecast:.2f} ({avg_change:+.1f}%)")

print("\n" + "="*80)
print("All models forecasted using Random Walk exogenous variables!")
print("="*80)