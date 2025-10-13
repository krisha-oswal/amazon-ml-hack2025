"""
STEP 4: Model Training & Submission Generation
Train ensemble models and create final predictions
Execution time: ~3-4 hours
"""
import ssl, certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("STEP 4: MODEL TRAINING & SUBMISSION")
print("="*70)

# SMAPE metric
def smape(y_true, y_pred):
    """Calculate SMAPE"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

# Load features
print("\nLoading features...")

# Try to load combined features (with images)
if Path('features/train_features_final.csv').exists():
    train = pd.read_csv('features/train_features_final.csv')
    test = pd.read_csv('features/test_features_final.csv')
    print("✅ Loaded combined (text + image) features")
else:
    # Fall back to text-only features
    train = pd.read_csv('features/train_features_text.csv')
    test = pd.read_csv('features/test_features_text.csv')
    print("✅ Loaded text-only features")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Prepare data
feature_cols = [col for col in train.columns if col not in ['sample_id', 'price']]
X = train[feature_cols].values
y = train['price'].values
X_test = test[feature_cols].values

print(f"\nFeatures: {len(feature_cols)}")
print(f"Training samples: {len(X):,}")
print(f"Test samples: {len(X_test):,}")
print(f"Price range: ${y.min():.2f} - ${y.max():,.2f}")
print(f"Price median: ${np.median(y):.2f}")

# Log transform target
y_log = np.log1p(y)
print(f"\nLog-transformed target range: {y_log.min():.4f} - {y_log.max():.4f}")

# Cross-validation setup
n_folds = 10
print(f"\nUsing {n_folds}-fold cross-validation")
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# ============================================================================
# MODEL 1: LIGHTGBM
# ============================================================================
print("\n" + "="*70)
print("MODEL 1: LIGHTGBM")
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
    'random_state': 42
}

lgb_oof = np.zeros(len(X))
lgb_test = np.zeros(len(X_test))
lgb_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}/{n_folds}")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_log[train_idx], y_log[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=5000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=500)
        ]
    )
    
    lgb_oof[val_idx] = model.predict(X_val)
    lgb_test += model.predict(X_test) / n_folds
    
    fold_smape = smape(np.expm1(y_val), np.expm1(lgb_oof[val_idx]))
    lgb_scores.append(fold_smape)
    print(f"Fold {fold + 1} SMAPE: {fold_smape:.4f}%")

lgb_cv = smape(y, np.expm1(lgb_oof))
print(f"\n{'='*50}")
print(f"LightGBM CV SMAPE: {lgb_cv:.4f}%")
print(f"Fold scores: {[f'{s:.4f}' for s in lgb_scores]}")
print(f"Std dev: {np.std(lgb_scores):.4f}%")
print(f"{'='*50}")

# ============================================================================
# MODEL 2: XGBOOST
# ============================================================================
print("\n" + "="*70)
print("MODEL 2: XGBOOST")
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
    'seed': 42,
    'tree_method': 'hist',
    'verbosity': 0
}

xgb_oof = np.zeros(len(X))
xgb_test = np.zeros(len(X_test))
xgb_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}/{n_folds}")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_log[train_idx], y_log[val_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=5000,
        evals=[(dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=500
    )
    
    xgb_oof[val_idx] = model.predict(dval)
    xgb_test += model.predict(dtest) / n_folds
    
    fold_smape = smape(np.expm1(y_val), np.expm1(xgb_oof[val_idx]))
    xgb_scores.append(fold_smape)
    print(f"Fold {fold + 1} SMAPE: {fold_smape:.4f}%")

xgb_cv = smape(y, np.expm1(xgb_oof))
print(f"\n{'='*50}")
print(f"XGBoost CV SMAPE: {xgb_cv:.4f}%")
print(f"Fold scores: {[f'{s:.4f}' for s in xgb_scores]}")
print(f"Std dev: {np.std(xgb_scores):.4f}%")
print(f"{'='*50}")

# ============================================================================
# MODEL 3: CATBOOST
# ============================================================================
print("\n" + "="*70)
print("MODEL 3: CATBOOST")
print("="*70)

cat_params = {
    'iterations': 5000,
    'learning_rate': 0.01,
    'depth': 8,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'verbose': 0,
    'early_stopping_rounds': 100,
    'task_type': 'CPU'
}

cat_oof = np.zeros(len(X))
cat_test = np.zeros(len(X_test))
cat_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}/{n_folds}")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_log[train_idx], y_log[val_idx]
    
    model = CatBoostRegressor(**cat_params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=500
    )
    
    cat_oof[val_idx] = model.predict(X_val)
    cat_test += model.predict(X_test) / n_folds
    
    fold_smape = smape(np.expm1(y_val), np.expm1(cat_oof[val_idx]))
    cat_scores.append(fold_smape)
    print(f"Fold {fold + 1} SMAPE: {fold_smape:.4f}%")

cat_cv = smape(y, np.expm1(cat_oof))
print(f"\n{'='*50}")
print(f"CatBoost CV SMAPE: {cat_cv:.4f}%")
print(f"Fold scores: {[f'{s:.4f}' for s in cat_scores]}")
print(f"Std dev: {np.std(cat_scores):.4f}%")
print(f"{'='*50}")

# ============================================================================
# ENSEMBLE
# ============================================================================
print("\n" + "="*70)
print("ENSEMBLE OPTIMIZATION")
print("="*70)

print("\nSearching for optimal weights...")
best_smape = float('inf')
best_weights = None

for w1 in np.arange(0.3, 0.7, 0.05):
    for w2 in np.arange(0.1, 0.5, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.1 or w3 > 0.5:
            continue
        
        ens_oof = w1 * lgb_oof + w2 * xgb_oof + w3 * cat_oof
        ens_smape = smape(y, np.expm1(ens_oof))
        
        if ens_smape < best_smape:
            best_smape = ens_smape
            best_weights = (w1, w2, w3)

w1, w2, w3 = best_weights
print(f"\nOptimal weights:")
print(f"  LightGBM: {w1:.3f}")
print(f"  XGBoost:  {w2:.3f}")
print(f"  CatBoost: {w3:.3f}")

# Final predictions
final_oof = w1 * lgb_oof + w2 * xgb_oof + w3 * cat_oof
final_test = w1 * lgb_test + w2 * xgb_test + w3 * cat_test

final_cv = smape(y, np.expm1(final_oof))

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"\nLightGBM CV SMAPE:  {lgb_cv:.4f}%")
print(f"XGBoost CV SMAPE:   {xgb_cv:.4f}%")
print(f"CatBoost CV SMAPE:  {cat_cv:.4f}%")
print(f"{'='*50}")
print(f"ENSEMBLE CV SMAPE:  {final_cv:.4f}%")
print(f"{'='*50}")

if final_cv < 10:
    print("\n🎉🎉🎉 OUTSTANDING! TARGET ACHIEVED! SMAPE < 10% 🎉🎉🎉")
elif final_cv < 12:
    print("\n🎉 EXCELLENT! SMAPE < 12%")
elif final_cv < 15:
    print("\n✅ VERY GOOD! SMAPE < 15%")
elif final_cv < 18:
    print("\n👍 GOOD! SMAPE < 18%")
else:
    print("\n⚠️  Needs improvement. Consider:")
    print("  - Adding image features (2-3% improvement)")
    print("  - Hyperparameter tuning")
    print("  - More feature engineering")

# ============================================================================
# CREATE SUBMISSION
# ============================================================================
print("\n" + "="*70)
print("CREATING SUBMISSION FILE")
print("="*70)

# Convert back to original scale
predictions = np.expm1(final_test)
predictions = np.maximum(predictions, 0.01)  # Ensure positive

submission = pd.DataFrame({
    'sample_id': test['sample_id'],
    'price': predictions
})

# Save submission
submission.to_csv('output/test_out.csv', index=False)

print(f"\n✅ Submission saved to: output/test_out.csv")
print(f"\nSubmission statistics:")
print(f"  Total predictions: {len(submission):,}")
print(f"  Price range: ${submission['price'].min():.2f} - ${submission['price'].max():,.2f}")
print(f"  Median price: ${submission['price'].median():.2f}")
print(f"  Mean price: ${submission['price'].mean():.2f}")

# Save OOF predictions for analysis
oof_df = pd.DataFrame({
    'sample_id': train['sample_id'],
    'actual_price': y,
    'predicted_price': np.expm1(final_oof),
    'lgb_pred': np.expm1(lgb_oof),
    'xgb_pred': np.expm1(xgb_oof),
    'cat_pred': np.expm1(cat_oof)
})
oof_df['error'] = oof_df['predicted_price'] - oof_df['actual_price']
oof_df['error_pct'] = (oof_df['error'] / oof_df['actual_price']) * 100

oof_df.to_csv('output/oof_predictions.csv', index=False)
print(f"✅ OOF predictions saved to: output/oof_predictions.csv")

# Save model performance summary
summary = pd.DataFrame([{
    'lgb_cv_smape': lgb_cv,
    'xgb_cv_smape': xgb_cv,
    'cat_cv_smape': cat_cv,
    'ensemble_cv_smape': final_cv,
    'lgb_weight': w1,
    'xgb_weight': w2,
    'cat_weight': w3,
    'n_folds': n_folds,
    'n_features': len(feature_cols)
}])
summary.to_csv('output/model_summary.csv', index=False)
print(f"✅ Model summary saved to: output/model_summary.csv")

print("\n" + "="*70)
print("STEP 4 COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  📄 output/test_out.csv (SUBMIT THIS)")
print("  📄 output/oof_predictions.csv")
print("  📄 output/model_summary.csv")

print("\nNext steps:")
print("  1. Verify submission format: python verify_submission.py")
print("  2. Update documentation with your CV SMAPE")
print("  3. Submit test_out.csv and documentation")

print(f"\n🎯 Your CV SMAPE: {final_cv:.4f}%")
print("Good luck! 🚀")