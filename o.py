"""
CPU-OPTIMIZED MODELING for MacBook Air
Lighter neural network + gradient boosting
Execution time: ~2-3 hours on CPU
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CPU-OPTIMIZED MODELING")
print("="*70)

# SMAPE metric
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

# Load features
print("\nLoading features...")
if Path('features/train_features_final.csv').exists():
    train = pd.read_csv('features/train_features_final.csv')
    test = pd.read_csv('features/test_features_final.csv')
    print("✅ Loaded combined (text + image) features")
else:
    train = pd.read_csv('features/train_features_text.csv')
    test = pd.read_csv('features/test_features_text.csv')
    print("✅ Loaded text-only features")

print(f"Train: {train.shape}")
print(f"Test: {test.shape}")

# Prepare data
feature_cols = [col for col in train.columns if col not in ['sample_id', 'price']]
X = train[feature_cols].values
y = train['price'].values
X_test = test[feature_cols].values

print(f"\nFeatures: {len(feature_cols)}")
print(f"Samples: {len(X):,}")
print(f"Price range: ${y.min():.2f} - ${y.max():,.2f}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Log transform target
y_log = np.log1p(y)

# ============================================================================
# CPU-OPTIMIZED NEURAL NETWORK
# ============================================================================

class LightweightNN(nn.Module):
    """Smaller network optimized for CPU"""
    
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        
        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 3
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 4
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            
            # Output
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)

class PriceDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

def train_nn(X_train, y_train, X_val, y_val):
    """Train neural network on CPU"""
    
    train_dataset = PriceDataset(X_train, y_train)
    val_dataset = PriceDataset(X_val, y_val)
    
    # Smaller batch size for CPU
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    model = LightweightNN(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(100):  # Reduced epochs for CPU
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: val_loss={val_loss:.6f}")
    
    model.load_state_dict(best_model)
    return model

# ============================================================================
# CROSS-VALIDATION
# ============================================================================

n_folds = 5
print(f"\nUsing {n_folds}-fold cross-validation")
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Storage
nn_oof = np.zeros(len(X))
nn_test = np.zeros(len(X_test))

lgb_oof = np.zeros(len(X))
lgb_test = np.zeros(len(X_test))

xgb_oof = np.zeros(len(X))
xgb_test = np.zeros(len(X_test))

cat_oof = np.zeros(len(X))
cat_test = np.zeros(len(X_test))

# ============================================================================
# NEURAL NETWORK
# ============================================================================
print("\n" + "="*70)
print("TRAINING NEURAL NETWORK (CPU)")
print("="*70)

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}/{n_folds}")
    
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y_log[train_idx], y_log[val_idx]
    
    model = train_nn(X_train, y_train, X_val, y_val)
    
    # Predictions
    model.eval()
    with torch.no_grad():
        val_tensor = torch.FloatTensor(X_val)
        nn_oof[val_idx] = model(val_tensor).squeeze().numpy()
        
        test_tensor = torch.FloatTensor(X_test_scaled)
        nn_test += model(test_tensor).squeeze().numpy() / n_folds
    
    fold_smape = smape(np.expm1(y_val), np.expm1(nn_oof[val_idx]))
    print(f"  ✅ Fold {fold + 1} SMAPE: {fold_smape:.4f}%")

nn_cv = smape(y, np.expm1(nn_oof))
print(f"\n{'='*50}")
print(f"Neural Network CV: {nn_cv:.4f}%")
print(f"{'='*50}")

# ============================================================================
# LIGHTGBM
# ============================================================================
print("\n" + "="*70)
print("TRAINING LIGHTGBM")
print("="*70)

lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 64,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'num_threads': 4,  # Use 4 CPU threads
    'random_state': 42
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}/{n_folds}...", end=' ')
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_log[train_idx], y_log[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(
        lgb_params, train_data,
        num_boost_round=3000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    lgb_oof[val_idx] = model.predict(X_val)
    lgb_test += model.predict(X_test) / n_folds
    
    fold_smape = smape(np.expm1(y_val), np.expm1(lgb_oof[val_idx]))
    print(f"SMAPE: {fold_smape:.4f}%")

lgb_cv = smape(y, np.expm1(lgb_oof))
print(f"LightGBM CV: {lgb_cv:.4f}%")

# ============================================================================
# XGBOOST
# ============================================================================
print("\n" + "="*70)
print("TRAINING XGBOOST")
print("="*70)

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'eta': 0.01,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'nthread': 4,  # Use 4 CPU threads
    'seed': 42
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}/{n_folds}...", end=' ')
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_log[train_idx], y_log[val_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    
    model = xgb.train(
        xgb_params, dtrain,
        num_boost_round=3000,
        evals=[(dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=0
    )
    
    xgb_oof[val_idx] = model.predict(dval)
    xgb_test += model.predict(dtest) / n_folds
    
    fold_smape = smape(np.expm1(y_val), np.expm1(xgb_oof[val_idx]))
    print(f"SMAPE: {fold_smape:.4f}%")

xgb_cv = smape(y, np.expm1(xgb_oof))
print(f"XGBoost CV: {xgb_cv:.4f}%")

# ============================================================================
# CATBOOST
# ============================================================================
print("\n" + "="*70)
print("TRAINING CATBOOST")
print("="*70)

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}/{n_folds}...", end=' ')
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_log[train_idx], y_log[val_idx]
    
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.01,
        depth=8,
        l2_leaf_reg=3,
        thread_count=4,  # Use 4 CPU threads
        random_seed=42,
        verbose=0,
        early_stopping_rounds=100
    )
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    cat_oof[val_idx] = model.predict(X_val)
    cat_test += model.predict(X_test) / n_folds
    
    fold_smape = smape(np.expm1(y_val), np.expm1(cat_oof[val_idx]))
    print(f"SMAPE: {fold_smape:.4f}%")

cat_cv = smape(y, np.expm1(cat_oof))
print(f"CatBoost CV: {cat_cv:.4f}%")

# ============================================================================
# ENSEMBLE
# ============================================================================
print("\n" + "="*70)
print("ENSEMBLE OPTIMIZATION")
print("="*70)

best_smape = float('inf')
best_weights = None

print("Searching for optimal weights...")
for w1 in np.arange(0.15, 0.35, 0.02):
    for w2 in np.arange(0.20, 0.35, 0.02):
        for w3 in np.arange(0.15, 0.30, 0.02):
            w4 = 1 - w1 - w2 - w3
            if w4 < 0.15 or w4 > 0.35:
                continue
            
            ens_oof = w1*nn_oof + w2*lgb_oof + w3*xgb_oof + w4*cat_oof
            ens_smape = smape(y, np.expm1(ens_oof))
            
            if ens_smape < best_smape:
                best_smape = ens_smape
                best_weights = (w1, w2, w3, w4)

w1, w2, w3, w4 = best_weights
print(f"\nOptimal weights:")
print(f"  Neural Net: {w1:.3f}")
print(f"  LightGBM:   {w2:.3f}")
print(f"  XGBoost:    {w3:.3f}")
print(f"  CatBoost:   {w4:.3f}")

final_oof = w1*nn_oof + w2*lgb_oof + w3*xgb_oof + w4*cat_oof
final_test = w1*nn_test + w2*lgb_test + w3*xgb_test + w4*cat_test

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"\nNeural Network:  {nn_cv:.4f}%")
print(f"LightGBM:        {lgb_cv:.4f}%")
print(f"XGBoost:         {xgb_cv:.4f}%")
print(f"CatBoost:        {cat_cv:.4f}%")
print(f"{'='*50}")
print(f"ENSEMBLE:        {best_smape:.4f}%")
print(f"{'='*50}")

if best_smape < 5:
    print("\n🎉🎉🎉 OUTSTANDING! You beat your friend! 🎉🎉🎉")
elif best_smape < 8:
    print("\n🎉 EXCELLENT! Very close to target!")
elif best_smape < 12:
    print("\n✅ GOOD! Getting there!")

# ============================================================================
# CREATE SUBMISSION
# ============================================================================
Path('output').mkdir(exist_ok=True)

predictions = np.expm1(final_test)
predictions = np.maximum(predictions, 0.01)

submission = pd.DataFrame({
    'sample_id': test['sample_id'],
    'price': predictions
})

submission.to_csv('output/test_out.csv', index=False)

print(f"\n✅ Submission saved: output/test_out.csv")
print(f"\nSubmission stats:")
print(f"  Min: ${submission['price'].min():.2f}")
print(f"  Max: ${submission['price'].max():.2f}")
print(f"  Median: ${submission['price'].median():.2f}")

print(f"\n🎯 Your CV SMAPE: {best_smape:.4f}%")
print("🚀 Ready to submit!")