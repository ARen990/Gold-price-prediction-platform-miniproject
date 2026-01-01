import os, sys, json
import pandas as pd
import altair as alt
from typing import Dict, List

# ---------------- Config defaults ----------------
SHORT_INPUTS_DEFAULT = {
    'Prophet':      os.path.join('forecasts', 'short', 'short_forecast_prophet.csv'),
    'SARIMAX':      os.path.join('forecasts', 'short', 'short_forecast_sarimax.csv'),
    'LSTM':         os.path.join('forecasts', 'short', 'short_forecast_lstm.csv'),
    'Bayesian VAR': os.path.join('forecasts', 'short', 'short_forecast_bayesian_var.csv'),
}
LONG_INPUTS_DEFAULT = {
    'Prophet':      os.path.join('forecasts', 'long', 'long_forecast_prophet.csv'),
    'SARIMAX':      os.path.join('forecasts', 'long', 'long_forecast_sarimax.csv'),
    'LSTM':         os.path.join('forecasts', 'long', 'long_forecast_lstm.csv'),
    'Bayesian VAR': os.path.join('forecasts', 'long', 'long_forecast_lstm.csv').replace('_lstm','_lstm'),
    'Bayesian VAR': os.path.join('forecasts', 'long', 'long_forecast_bayesian_var.csv'),
}

SCRIPT_DIR  = os.path.dirname(os.path.abspath(sys.argv[0]))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
HIST_BASE   = os.getenv('HIST_BASE_DIR', os.path.join(PROJECT_DIR, 'results'))

HIST_SHORT_CAND = ['./web/static/chart/short_historic.csv']
HIST_LONG_CAND  = ['./web/static/chart/long_historic.csv']

# ---------------- Utility ----------------
def _smart_datetime(s):
    dt = pd.to_datetime(s, errors='coerce', utc=True, infer_datetime_format=True)
    if dt.isna().all():
        num = pd.to_numeric(s, errors='coerce')
        if num.notna().any():
            dt = pd.to_datetime(num, unit='ms' if num.dropna().median()>1e11 else 's',
                                errors='coerce', utc=True)
    try: dt = dt.tz_convert(None)
    except Exception: pass
    return dt

def _load_any_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lstrip('\ufeff') for c in df.columns]
    # date col
    dcands = ['Date','date','Datetime','datetime','timestamp','time','Time','ds','Index','index','Unnamed: 0']
    dcol = next((c for c in dcands if c in df.columns), None)
    if dcol is None:
        df = df.reset_index()
        dcol = df.columns[0]
    # value col
    vcands = ['GC=F_close','Close','close','Adj Close','adj_close','Price','price','yhat','value','Value']
    vcol = next((c for c in vcands if c in df.columns), None)
    if vcol is None:
        close_like = [c for c in df.columns if 'close' in c.lower()]
        vcol = close_like[0] if close_like else [c for c in df.columns if c != dcol][0]
    out = df[[dcol, vcol]].copy()
    out.columns = ['Date','Value']
    out['Date']  = _smart_datetime(out['Date'])
    out['Value'] = pd.to_numeric(out['Value'], errors='coerce')
    out.dropna(subset=['Date','Value'], inplace=True)
    return out.sort_values('Date').reset_index(drop=True)

def _pick_existing(base_dir: str, names: List[str]) -> str | None:
    for n in names:
        p = os.path.join(base_dir, n)
        if os.path.exists(p): return p
    return None

def _load_historical_full(hist_path: str | None) -> pd.DataFrame:
    # Try explicit path, then common places
    search = []
    if hist_path: search.append(hist_path)
    for base in [HIST_BASE, os.path.join(PROJECT_DIR,'results'), os.path.join(SCRIPT_DIR,'results'), '.']:
        for n in set(HIST_LONG_CAND + HIST_SHORT_CAND):
            search.append(os.path.join(base, n))
    for p in search:
        if p and os.path.exists(p):
            try:
                h = _load_any_csv(p).rename(columns={'Value':'Price'})
                if not h.empty:
                    h['Model'] = 'Historical'; h['Kind'] = 'Historical'
                    print(f"[OK] Historical from {p} rows={len(h)}")
                    return h
            except Exception as e:
                print(f"[WARN] cannot read historic {p}: {e}")
    raise RuntimeError("[FATAL] Historical not found. Set HIST_BASE_DIR or place *_historic.csv / *_ohlc.csv under ./results or ./static/chart")

def _load_predict_frames(inputs_map: Dict[str,str]) -> List[pd.DataFrame]:
    frames = []
    for model, path in inputs_map.items():
        if not os.path.exists(path):
            print(f"[WARN] missing: {model} -> {path}")
            continue
        d = _load_any_csv(path).rename(columns={'Value':'Price'})
        d['Model'] = model; d['Kind'] = 'Predict'
        frames.append(d)
        print(f"[OK] Predict {model} rows={len(d)} start={d['Date'].iloc[0] if len(d) else 'NA'}")
    if not frames:
        raise RuntimeError("[FATAL] no predict data loaded")
    return frames

SPANS = {
    '1H':  pd.Timedelta(hours=1),
    '24H': pd.Timedelta(hours=24),
    '1W':  pd.Timedelta(days=7),
    '1M':  pd.Timedelta(days=30),
    '3M':  pd.Timedelta(days=90),
    '6M':  pd.Timedelta(days=180),
}

def _cut_model_from_its_start(df_model: pd.DataFrame, period_key: str) -> pd.DataFrame:
    start = df_model['Date'].min()
    end   = start + SPANS.get(period_key, pd.Timedelta(days=180))
    cut   = df_model[(df_model['Date'] >= start) & (df_model['Date'] <= end)].copy()
    print(f"[CUT] {df_model['Model'].iloc[0]} {period_key} start={start} end={end} rows={len(cut)}")
    return cut

def _make_spec(df: pd.DataFrame, *, title: str, is_long: bool) -> dict:
    palette = {'Historical':"#C1CEC8",'Prophet':'#1f77b4','SARIMAX':'#ff7f0e','LSTM':'#2ca02c','Bayesian VAR':'#d62728'}
    uniq = list(pd.unique(df['Model']))
    domain, rng = [], []
    if 'Historical' in uniq: domain.append('Historical'); rng.append(palette['Historical'])
    for m in ['Prophet','SARIMAX','LSTM','Bayesian VAR']:
        if m in uniq: domain.append(m); rng.append(palette[m])

    xfmt = '%Y-%m-%d' if is_long else '%Y-%m-%d %H:%M'
    chart = alt.Chart(df).mark_line(point={'filled': True, 'size': 30} if not is_long else False).encode(
        x=alt.X('Date:T', title='Date',
                axis=alt.Axis(format=xfmt, labelAngle=-45, tickCount=8)),
        y=alt.Y('Price:Q', title='Gold Price (USD)'),
        color=alt.Color('Model:N', title='Series', scale=alt.Scale(domain=domain, range=rng)),
        strokeDash=alt.condition(alt.datum.Model=='Historical', alt.value([5,5]), alt.value([0])),
        opacity=alt.condition(alt.datum.Model=='Historical', alt.value(0.7), alt.value(1.0)),
        tooltip=[
            alt.Tooltip('Date:T',  title='Date',  format=xfmt),
            alt.Tooltip('Price:Q', title='Price', format=',.2f'),
            alt.Tooltip('Model:N', title='Series'),
            alt.Tooltip('Kind:N',  title='Kind'),
        ]
    ).properties(title=title).interactive()

    spec = chart.to_dict()
    spec["autosize"] = {"type": "fit", "contains": "padding"}
    spec["width"] = "container"; spec["height"] = "container"; spec["background"] = "#333333"
    spec.setdefault('config', {})
    spec['config']['view']={'continuousWidth':400,'continuousHeight':300}
    spec['config']['axis']={'labelColor':'#ecf0f1', 'titleColor':'#ecf0f1',
          'gridColor':'#34495e',  'domainColor':'#7f8c8d', 'tickColor':'#7f8c8d'
        }
    spec['config']['legend']={'labelColor':'#ecf0f1','titleColor':'#ecf0f1'}
    spec['config']['title']={'color':'#ecf0f1'}
    return spec

def _save_json(spec: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)
    print(f"[SAVE] {out_path}")

# ---------------- Public API ----------------
def generate_all_jsons(out_dir='static/chart',
                       short_inputs: Dict[str,str] | None = None,
                       long_inputs:  Dict[str,str] | None = None,
                       hist_base_dir: str | None = None):
    """Build 6 JSON files for All: 1H/24H/1W/1M/3M/6M."""
    global HIST_BASE
    if short_inputs is None: short_inputs = SHORT_INPUTS_DEFAULT
    if long_inputs  is None: long_inputs  = LONG_INPUTS_DEFAULT
    if hist_base_dir: HIST_BASE = hist_base_dir

    os.makedirs(out_dir, exist_ok=True)
    hist_short = _pick_existing(HIST_BASE, HIST_SHORT_CAND)
    hist_long  = _pick_existing(HIST_BASE, HIST_LONG_CAND)

    targets = [
        ('1H',  False, False),
        ('24H', False, False),
        ('1W',  False, False),
        ('1M',  True,  True ),
        ('3M',  True,  True ),
        ('6M',  True,  True ),
    ]
    for key, use_long, is_long in targets:
        inputs = long_inputs if use_long else short_inputs
        hist   = _load_historical_full(hist_long if use_long else hist_short)
        preds  = _load_predict_frames(inputs)
        sliced = [_cut_model_from_its_start(df, key) for df in preds]
        preds_concat = pd.concat(sliced, ignore_index=True) if sliced else pd.DataFrame(columns=['Date','Price','Model','Kind'])
        combined = pd.concat([hist, preds_concat], ignore_index=True).sort_values('Date')
        spec = _make_spec(combined, title="", is_long=is_long)
        _save_json(spec, os.path.join(out_dir, f"forecast_all_{key}.json"))

# ---- Legacy shim so old code doesn't break (optional) ----
def build_all_models_line(period: str, inputs_map: Dict[str,str], output_json_path: str,
                          historical_csv_path: str | None = None, rows_to_keep: int | None = None, title_suffix: str = ""):
    """Single JSON builder (kept for backward compatibility)."""
    is_long = (period.lower() == 'long')
    hist = _load_historical_full(historical_csv_path)
    preds = _load_predict_frames(inputs_map)
    if rows_to_keep is not None:
        preds = [df.iloc[:rows_to_keep].copy() for df in preds]
    all_df = pd.concat([hist] + preds, ignore_index=True).sort_values('Date')
    spec = _make_spec(all_df, title="".strip(), is_long=is_long)
    _save_json(spec, output_json_path)

# ---------------- CLI ----------------
if __name__ == '__main__':
    generate_all_jsons(out_dir=os.path.join('static','chart'))
