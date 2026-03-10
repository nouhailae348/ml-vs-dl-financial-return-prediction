import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CSV_FILE       = 'clean_multi_asset_dataset.csv'
TEST_SIZE      = 0.2
LSTM_TIMESTEPS = 10
EPOCHS         = 100
BATCH_SIZE     = 64

df = pd.read_csv(CSV_FILE)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Asset_ID', 'Date']).reset_index(drop=True)

print("=" * 55)
print("  DATASET OVERVIEW")
print("=" * 55)
print(f"  Shape         : {df.shape}")
print(f"  Date range    : {df['Date'].min().date()} -> {df['Date'].max().date()}")
print(f"  Assets        : {df['Asset_ID'].unique().tolist()}")
print(f"  Columns       : {df.columns.tolist()}")
print("=" * 55)

def engineer_features(df):
    g = df.groupby('Asset_ID')
    df['Return'] = g['Close'].pct_change()
    # lag features so we don't leak future data into the model
    for lag in [1, 2, 3, 5]:
        df[f'Return_lag{lag}'] = g['Return'].shift(lag)
    df['Return_roll_mean5']  = g['Return'].transform(lambda x: x.shift(1).rolling(5).mean())
    df['Return_roll_std5']   = g['Return'].transform(lambda x: x.shift(1).rolling(5).std())
    df['Return_roll_mean10'] = g['Return'].transform(lambda x: x.shift(1).rolling(10).mean())
    df['Close_lag1']         = g['Close'].shift(1)
    df['High_Low_range']     = (df['High'] - df['Low']) / df['Close']
    df['Open_Close_gap']     = (df['Close'] - df['Open']) / df['Open']
    df['High_lag1']          = g['High'].shift(1)
    df['Low_lag1']           = g['Low'].shift(1)
    df['Volume_lag1']        = g['Volume'].shift(1)
    df['Volume_roll_mean5']  = g['Volume'].transform(lambda x: x.shift(1).rolling(5).mean())
    df['Volume_ratio']       = df['Volume_lag1'] / df['Volume_roll_mean5']
    df['Momentum_5']         = g['Close'].transform(lambda x: x.shift(1) / x.shift(6) - 1)
    df['Momentum_10']        = g['Close'].transform(lambda x: x.shift(1) / x.shift(11) - 1)
    df['Volatility_5']       = g['Return'].transform(lambda x: x.shift(1).rolling(5).std())
    df['Volatility_10']      = g['Return'].transform(lambda x: x.shift(1).rolling(10).std())
    return df

df = engineer_features(df)
df.dropna(inplace=True)
print(f"\n  Shape after feature engineering : {df.shape}")

FEATURES = [
    'Return_lag1', 'Return_lag2', 'Return_lag3', 'Return_lag5',
    'Return_roll_mean5', 'Return_roll_std5', 'Return_roll_mean10',
    'Close_lag1', 'High_Low_range', 'Open_Close_gap',
    'High_lag1', 'Low_lag1',
    'Volume_lag1', 'Volume_roll_mean5', 'Volume_ratio',
    'Momentum_5', 'Momentum_10',
    'Volatility_5', 'Volatility_10',
    'Asset_ID_encoded'
]
TARGET = 'Return'

X = df[FEATURES].values
y = df[TARGET].values

# keep the split time-ordered so we don't accidentally train on future data
split_idx   = int(len(X) * (1 - TEST_SIZE))
X_train_raw = X[:split_idx]
X_test_raw  = X[split_idx:]
y_train_raw = y[:split_idx]
y_test_raw  = y[split_idx:]

print(f"  Train samples : {len(X_train_raw)}")
print(f"  Test  samples : {len(X_test_raw)}")

# scale features using only training data to avoid data leakage
feat_scaler    = StandardScaler()
X_train_scaled = feat_scaler.fit_transform(X_train_raw)
X_test_scaled  = feat_scaler.transform(X_test_raw)

tgt_scaler     = StandardScaler()
y_train_scaled = tgt_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()

# LSTM needs 3D input: (samples, timesteps, features)
def make_sequences(X, y, timesteps):
    Xs, ys = [], []
    for i in range(timesteps, len(X)):
        Xs.append(X[i - timesteps:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = make_sequences(X_train_scaled, y_train_scaled, LSTM_TIMESTEPS)
X_test_seq,  y_test_seq  = make_sequences(X_test_scaled,  y_test_raw,     LSTM_TIMESTEPS)

# flat arrays for the non-sequence models, aligned to the same length as sequences
X_train_flat    = X_train_scaled[LSTM_TIMESTEPS:]
X_test_flat     = X_test_scaled[LSTM_TIMESTEPS:]
y_train_flat    = y_train_raw[LSTM_TIMESTEPS:]
y_test_flat     = y_test_raw[LSTM_TIMESTEPS:]
y_train_flat_sc = y_train_scaled[LSTM_TIMESTEPS:]

n_features = X_train_scaled.shape[1]

def build_ann(n_features):
    model = Sequential([
        Input(shape=(n_features,)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_lstm(timesteps, n_features):
    model = Sequential([
        Input(shape=(timesteps, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

early_stop = EarlyStopping(monitor='val_loss', patience=10,
                            restore_best_weights=True, verbose=0)

print("\n" + "=" * 55)
print("  MODEL TRAINING & EVALUATION")
print("=" * 55)

print("\n  > Linear Regression")
lr_model = LinearRegression()
lr_model.fit(X_train_flat, y_train_flat)
y_pred_lr = lr_model.predict(X_test_flat)

print("  > Random Forest")
rf_model = RandomForestRegressor(
    n_estimators=300, max_depth=10,
    min_samples_leaf=5, random_state=42, n_jobs=-1)
rf_model.fit(X_train_flat, y_train_flat)
y_pred_rf = rf_model.predict(X_test_flat)

print("  > ANN (Artificial Neural Network)")
ann_model   = build_ann(n_features)
ann_history = ann_model.fit(
    X_train_flat, y_train_flat_sc,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=0
)
# inverse transform to get predictions back in original return scale
y_pred_ann = tgt_scaler.inverse_transform(
    ann_model.predict(X_test_flat, verbose=0)).flatten()
print(f"     Stopped at epoch {len(ann_history.history['loss'])}")

print("  > LSTM (Long Short-Term Memory)")
lstm_model   = build_lstm(LSTM_TIMESTEPS, n_features)
lstm_history = lstm_model.fit(
    X_train_seq, y_train_seq,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=0
)
y_pred_lstm = tgt_scaler.inverse_transform(
    lstm_model.predict(X_test_seq, verbose=0)).flatten()
print(f"     Stopped at epoch {len(lstm_history.history['loss'])}")

model_preds = {
    'Linear Regression': (y_test_flat, y_pred_lr),
    'Random Forest':     (y_test_flat, y_pred_rf),
    'ANN':               (y_test_flat, y_pred_ann),
    'LSTM':              (y_test_seq,  y_pred_lstm),
}

results = {}
for name, (y_true, y_pred) in model_preds.items():
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    print(f"\n  [{name}]")
    print(f"     MAE  : {mae:.6f}")
    print(f"     RMSE : {rmse:.6f}")
    print(f"     R2   : {r2:.6f}")

print("\n" + "=" * 55)
print("  PER-ASSET BREAKDOWN")
print("=" * 55)

test_df = df.iloc[split_idx + LSTM_TIMESTEPS:].copy().reset_index(drop=True)
test_df['Pred_Linear Regression'] = y_pred_lr
test_df['Pred_Random Forest']     = y_pred_rf
test_df['Pred_ANN']               = y_pred_ann
test_df['Pred_LSTM']              = y_pred_lstm

per_asset_results = {}
model_names = list(model_preds.keys())

for name in model_names:
    print(f"\n  [{name}]")
    asset_metrics = {}
    for asset in test_df['Asset_ID'].unique():
        mask   = test_df['Asset_ID'] == asset
        actual = test_df.loc[mask, 'Return']
        pred   = test_df.loc[mask, f'Pred_{name}']
        r2_a   = r2_score(actual, pred)
        rmse_a = np.sqrt(mean_squared_error(actual, pred))
        asset_metrics[asset] = {'R2': r2_a, 'RMSE': rmse_a}
        print(f"     {asset:>6}  ->  R2: {r2_a:.4f}  |  RMSE: {rmse_a:.6f}")
    per_asset_results[name] = asset_metrics

metrics_df = pd.DataFrame(results).T.round(6)
metrics_df.to_csv('model_metrics.csv')
print(f"\n  Metrics saved -> model_metrics.csv")

COLORS = {
    'Linear Regression': '#2196F3',
    'Random Forest':     '#E53935',
    'ANN':               '#43A047',
    'LSTM':              '#FB8C00'
}
N_PLOT = 300

for name in model_names:
    y_true, y_pred = model_preds[name]
    color = COLORS[name]
    n = min(N_PLOT, len(y_true))

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f'Return Prediction - {name}', fontsize=15, fontweight='bold')

    ax.plot(range(n), y_true[:n], color='black', alpha=0.55, linewidth=1, label='Actual')
    ax.plot(range(n), y_pred[:n], color=color,   alpha=0.85, linewidth=1, label='Predicted')
    ax.set_title('Actual vs Predicted', fontsize=12)
    ax.set_xlabel('Test Sample Index'); ax.set_ylabel('Return')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'comparison_{name.lower().replace(" ", "_")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Plot saved -> {fname}")

# put all 4 scatter plots in one figure for easy comparison
fig, axes_sc = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Scatter Plots - All Models', fontsize=15, fontweight='bold')

for ax, name in zip(axes_sc.flatten(), model_names):
    y_true, y_pred = model_preds[name]
    color = COLORS[name]
    ax.scatter(y_true, y_pred, alpha=0.2, s=6, color=color)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', linewidth=1.2, label='Perfect fit')
    ax.set_title(f'{name}  (R2={results[name]["R2"]:.4f})', fontsize=12)
    ax.set_xlabel('Actual Return'); ax.set_ylabel('Predicted Return')
    ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scatter_all_models.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Plot saved -> scatter_all_models.png")

fig, axes2 = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Diagnostics', fontsize=15, fontweight='bold')

ax = axes2[0]
for name, (y_true, y_pred) in model_preds.items():
    ax.hist(y_true - y_pred, bins=70, alpha=0.45, label=name, color=COLORS[name])
ax.axvline(0, color='black', linewidth=1.2, linestyle='--')
ax.set_title('Residuals Distribution'); ax.set_xlabel('Residual'); ax.set_ylabel('Frequency')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes2[1]
rf_imp = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values()
rf_imp.plot(kind='barh', ax=ax, color='#E53935', alpha=0.8)
ax.set_title('Random Forest - Feature Importances'); ax.set_xlabel('Importance Score')
ax.grid(True, alpha=0.3)

ax = axes2[2]
ax.plot(ann_history.history['loss'],      color=COLORS['ANN'],  label='ANN Train')
ax.plot(ann_history.history['val_loss'],  color=COLORS['ANN'],  linestyle='--', label='ANN Val')
ax.plot(lstm_history.history['loss'],     color=COLORS['LSTM'], label='LSTM Train')
ax.plot(lstm_history.history['val_loss'], color=COLORS['LSTM'], linestyle='--', label='LSTM Val')
ax.set_title('Training Loss Curves (ANN & LSTM)')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Plot saved -> diagnostics.png")

assets   = test_df['Asset_ID'].unique()
n_assets = len(assets)
n_models = len(model_names)

fig, axes3 = plt.subplots(n_assets, n_models,
                           figsize=(6 * n_models, 5 * n_assets),
                           squeeze=False)
fig.suptitle('Per-Asset Return Prediction (Test Set)', fontsize=16, fontweight='bold')

for row, asset in enumerate(assets):
    mask   = test_df['Asset_ID'] == asset
    actual = test_df.loc[mask, 'Return'].values
    idx    = np.arange(len(actual))

    for col, name in enumerate(model_names):
        ax     = axes3[row, col]
        y_pred = test_df.loc[mask, f'Pred_{name}'].values
        r2_val = per_asset_results[name][asset]['R2']
        ax.plot(idx, actual, color='black',      alpha=0.55, linewidth=1, label='Actual')
        ax.plot(idx, y_pred, color=COLORS[name], alpha=0.85, linewidth=1, label='Predicted')
        ax.set_title(f'{asset}  |  {name}  (R2={r2_val:.4f})', fontsize=10)
        ax.set_xlabel('Test Sample Index'); ax.set_ylabel('Return')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('per_asset_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Plot saved -> per_asset_predictions.png")

fig, ax = plt.subplots(figsize=(14, 5))
x     = np.arange(n_assets)
width = 0.2

all_r2_vals = [per_asset_results[m][a]['R2'] for m in model_names for a in assets]
y_min = min(0, min(all_r2_vals)) - 0.05

for i, name in enumerate(model_names):
    r2_vals = [per_asset_results[name][a]['R2'] for a in assets]
    bars = ax.bar(x + i * width, r2_vals, width, label=name,
                  color=COLORS[name], alpha=0.8)
    for bar, val in zip(bars, r2_vals):
        va  = 'bottom' if val >= 0 else 'top'
        ypos = bar.get_height() + 0.005 if val >= 0 else bar.get_height() - 0.005
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f'{val:.3f}', ha='center', va=va, fontsize=8)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(assets, fontsize=12)
ax.set_title('R2 Score per Asset - All Models', fontsize=14, fontweight='bold')
ax.set_ylabel('R2 Score'); ax.set_ylim(bottom=y_min)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.legend(); ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('r2_per_asset.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Plot saved -> r2_per_asset.png")

print("\n" + "=" * 55)
print("  ALL DONE")
print("  Files generated:")
print("    - model_metrics.csv")
print("    - comparison_linear_regression.png")
print("    - comparison_random_forest.png")
print("    - comparison_ann.png")
print("    - comparison_lstm.png")
print("    - scatter_all_models.png")
print("    - diagnostics.png")
print("    - per_asset_predictions.png")
print("    - r2_per_asset.png")
print("=" * 55)