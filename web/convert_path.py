import json
import pandas as pd
import altair as alt
import os
import sys

def choose(*names):
    """เลือกคอลัมน์ตามชื่อ (ไม่แคสเซนสิทีฟ) พร้อม fallback แบบ contains"""
    for n in names:
        c = lower.get(n)
        if c:
            return c
    # fallback: ค้นหาแบบ contains คำหลักตัวสุดท้าย เช่น 'close','high','low','open'
    key = names[0].split('_')[-1]
    for c in df.columns:
        if key in c.lower():
            return c
    return None

def generate_forecast_chart(
    model_name,
    input_filename, 
    output_dir='static/chart', 
):
    # --- resolve paths ---
    try:
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    except IndexError:
        script_dir = os.path.abspath(os.path.dirname(__file__))

    output_folder_path = os.path.join(script_dir, output_dir)
    input_path = input_filename if os.path.isabs(input_filename) else os.path.join(script_dir, input_filename)
    output_filename = f'forecast_{model_name.lower()}_candlestick.json'
    output_path = os.path.join(output_folder_path, output_filename)

    print(f"\n--- Chart Generation Status for {model_name} ---")
    print(f"CSV: {input_path}")
    print(f"OUT: {output_path}")

    os.makedirs(output_folder_path, exist_ok=True)

    # --- load csv (force header with Date) ---
    try:
        # ไฟล์ช่วงสั้นของคุณมีหัว 3 คอลัมน์ (GC=F_*) แต่ข้อมูลจริงมี 4 ช่อง (เวลาคือคอลัมน์แรก)
        # บังคับชื่อคอลัมน์ให้ครบ 4 ตัว แล้ว parse Date ทันที
        df = pd.read_csv(
            input_path,
            header=0,
            names=['Date', 'GC=F_close', 'GC=F_high', 'GC=F_low'],
            parse_dates=['Date']
        )
    except FileNotFoundError:
        print(f"FATAL ERROR: The input CSV file '{input_filename}' was not found.")
        print(f"Please ensure it is located at: {input_path}")
        return

    # --- columns (GC=F_* first, fallback Close/High/Low) ---
    # --- normalize columns to Date/Open/High/Low/Close ---
    df.rename(columns={
        'GC=F_close': 'Close',
        'GC=F_high':  'High',
        'GC=F_low':   'Low',
    }, inplace=True)

    # ถ้าไม่มี Open → สร้างจาก Close.shift(1)
    import numpy as np
    if 'Open' not in df.columns:
        df['Open'] = pd.to_numeric(df['Close'], errors='coerce').shift(1)

    # บังคับ numeric และล้าง NaN/Inf
    for c in ['Open', 'High', 'Low', 'Close']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['Date', 'Open', 'High', 'Low', 'Close'], inplace=True)


    # --- detect horizon for axis/time formatting ---
    is_short = str(model_name).lower().startswith('short_')
    # Short: keep full datetime (with time); Long: date only
    df_ohlc = df[['Date', 'Close', 'High', 'Low']].copy()
    df_ohlc['Date'] = pd.to_datetime(df_ohlc['Date'], errors='coerce')
    df_ohlc.dropna(subset=['Date'], inplace=True)

    # Prepare a display field (string) only for axis/tooltip formatting
    if is_short:
        # Keep intraday resolution; don’t downcast to date only
        df_ohlc['DateStr'] = df_ohlc['Date'].dt.strftime('%Y-%m-%d %H:%M')
        axis_time_format = '%Y-%m-%d %H:%M'
    else:
        # Long horizon → date only
        df_ohlc['DateStr'] = df_ohlc['Date'].dt.strftime('%Y-%m-%d')
        axis_time_format = '%Y-%m-%d'

    # Open = previous Close
    df_ohlc['Open'] = pd.to_numeric(df_ohlc['Close'], errors='coerce').shift(1)
    for col in ['Close', 'High', 'Low', 'Open']:
        df_ohlc[col] = pd.to_numeric(df_ohlc[col], errors='coerce')
    df_ohlc = df_ohlc.set_index('Date').sort_index()

    if not is_short:
        # build a daily calendar and forward-fill values
        full_days = pd.date_range(df_ohlc.index.min(), df_ohlc.index.max(), freq='D')
        df_ohlc = df_ohlc.reindex(full_days).ffill()

    # (re)compute OHLC so candles render correctly
    df_ohlc['Open'] = df_ohlc['Close'].shift(1)
    # take max/min across available columns to keep wicks sensible
    df_ohlc['High'] = df_ohlc[['Open', 'Close', 'High']].max(axis=1)
    df_ohlc['Low']  = df_ohlc[['Open', 'Close', 'Low']].min(axis=1)

    # clean and bring Date back as a column
    df_ohlc.dropna(subset=['Open', 'Close', 'High', 'Low'], inplace=True)
    df_ohlc = df_ohlc.reset_index().rename(columns={'index': 'Date'})

    # (re)build DateStr after reset_index
    if is_short:
        df_ohlc['DateStr'] = df_ohlc['Date'].dt.strftime('%Y-%m-%d %H:%M')
    else:
        df_ohlc['DateStr'] = df_ohlc['Date'].dt.strftime('%Y-%m-%d')
        
    if df_ohlc.empty:
        print("FATAL ERROR: Dataframe is empty after processing. No data to plot.")
        return

    df_ohlc['Price_Change']  = df_ohlc['Close'] - df_ohlc['Open']
    df_ohlc['Change_Percent'] = (df_ohlc['Price_Change'] / df_ohlc['Open']) * 100
    df_ohlc['Color'] = df_ohlc['Price_Change'].apply(lambda x: 'Up' if float(x) >= 0 else 'Down')

    # --- selections: unique names per model & add_params at layer only ---
    # brush_name = f"brush_{model_name.lower()}"
    zoom_name  = f"zoom_{model_name.lower()}"

    # ===== FONT SIZES =====
    TITLE_FS = 20
    AXIS_LABEL_FS = 18   # ขนาดตัวเลขแกน (รวมแกน X)
    AXIS_TITLE_FS = 18   # ชื่อแกน
    # ======================

    base = alt.Chart(df_ohlc).properties(
        # title=f'Gold Price Forecast ({model_name} Model)',
        # title=f'Gold Price Forecast',
        width='container',
        height=400
    )

    # x uses the real temporal field (Date:T) so Vega-Lite knows the scale is time.
    # axis label/tooltip use DateStr for a nicer formatted string (with time for short).
    x_enc = alt.X(
        'Date:T',
        title='Date' if not is_short else 'Date / Time',
        axis=alt.Axis(
            format=axis_time_format,
            labelAngle=-60,            # อ่านง่ายขึ้นเมื่อฟอนต์ใหญ่
            labelFontSize=AXIS_LABEL_FS,
            titleFontSize=AXIS_TITLE_FS
        ),
    )

    rule = base.mark_rule().encode(
        x=x_enc,
        y=alt.Y('Low:Q', title='Gold Price (USD)', scale=alt.Scale(zero=False)),
        y2='High:Q',
        color=alt.Color('Color:N', 
                        scale=alt.Scale(domain=['Up', 'Down'], range=['#2ecc71', '#e74c3c']),
                        legend=None),
        tooltip=[
            # show the pre-formatted DateStr in tooltip (has time for short)
            alt.Tooltip('DateStr:N', title='Date' if not is_short else 'Date / Time'),
            alt.Tooltip('Open:Q', title='Open', format='$.2f'),
            alt.Tooltip('Close:Q', title='Close', format='$.2f'),
            alt.Tooltip('High:Q', title='High', format='$.2f'),
            alt.Tooltip('Low:Q', title='Low', format='$.2f'),
            alt.Tooltip('Price_Change:Q', title='Change', format='$.2f'),
            alt.Tooltip('Change_Percent:Q', title='Change %', format='.2f')
        ]
    )

    bar = base.mark_bar(size=10).encode(
        x=x_enc,
        y='Open:Q',
        y2='Close:Q',
        color=alt.Color('Color:N', 
                        scale=alt.Scale(domain=['Up', 'Down'], range=['#2ecc71', '#e74c3c']),
                        legend=alt.Legend(title='Price Direction'))
    )

    # Selections (define once)
    # brush = alt.selection_interval(name=brush_name, encodings=['x'])
    zoom_pan = alt.selection_interval(name=zoom_name, bind='scales', encodings=['x'])

    line = base.mark_line(color="#00FFFF", strokeWidth=2).encode(
        x=x_enc,
        y='Close:Q',
        tooltip=[
            alt.Tooltip('DateStr:N', title='Date' if not is_short else 'Date / Time'),
            alt.Tooltip('Close:Q', title='Close Price', format='$.2f')
        ]
    )

    points = base.mark_point(filled=True, size=1000, opacity=0).encode(
        x=x_enc,
        y='Close:Q',
        tooltip=[
            alt.Tooltip('DateStr:N', title='Date' if not is_short else 'Date / Time'),
            alt.Tooltip('Open:Q', title='Open', format='$.2f'),
            alt.Tooltip('Close:Q', title='Close', format='$.2f'),
            alt.Tooltip('High:Q', title='High', format='$.2f'),
            alt.Tooltip('Low:Q', title='Low', format='$.2f'),
            alt.Tooltip('Price_Change:Q', title='Change', format='+.2f'),
            alt.Tooltip('Change_Percent:Q', title='Change %', format='+.2f')
        ]
    )

    chart = alt.layer(rule, bar, line, points).add_params(zoom_pan).configure_view(
        stroke=None
    ).configure_axis(
        gridColor='#34495e',
        domainColor='#7f8c8d',
        tickColor='#7f8c8d',
        labelColor='#ecf0f1',
        labelFontSize=100,   
        titleFontSize=100   
    ).configure_title(
        color='#ecf0f1',
        fontSize=20
    ).configure_legend(
        labelColor='#ecf0f1',
        titleColor='#ecf0f1'
    )

    spec = chart.to_dict()

    # page-friendly styling
    spec['background'] = "#333333"
    spec['autosize'] = {"type": "fit", "contains": "padding"}
    spec["width"] = "container"
    spec["height"] = "container"
    if 'config' not in spec:
        spec['config'] = {}
    spec['config']['view'] = {'continuousWidth': 400, 'continuousHeight': 300}
    spec['config']['axis'] = {'labelFontSize': 12, 'titleFontSize': 14}

    # ALWAYS overwrite (atomic)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_path)

    print(f"SUCCESS: Interactive chart specification saved to '{output_path}'.")
    print("---------------------------------------")


if __name__ == '__main__':
    # Example mapping (แก้ให้ตรงไฟล์จริงของคุณ)
    models_to_plot = {
        # long
        'long_prophet':       'forecasts/long/long_forecast_prophet.csv',
        'long_sarimax':       'forecasts/long/long_forecast_sarimax.csv',
        'long_lstm':          'forecasts/long/long_forecast_lstm.csv',
        'long_bayesian_var':  'forecasts/long/long_forecast_bayesian_var.csv',
        # short
        'short_prophet':      'forecasts/short/short_forecast_prophet.csv',
        'short_sarimax':      'forecasts/short/short_forecast_sarimax.csv',
        'short_lstm':         'forecasts/short/short_forecast_lstm.csv',
        'short_bayesian_var': 'forecasts/short/short_forecast_bayesian_var.csv',
    }

    for model_name, filename in models_to_plot.items():
        generate_forecast_chart(model_name, filename)

    print("\n✅ All interactive candlestick charts have been generated.")