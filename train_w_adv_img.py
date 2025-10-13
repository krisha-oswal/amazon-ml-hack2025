"""
TRAINING WITH ADVANCED IMAGE FEATURES
Uses the enhanced image features for better performance
Expected SMAPE: 12-16%
"""

import pandas as pd
import numpy as np
import re
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

print("="*70)
print("TRAINING WITH ADVANCED IMAGE FEATURES")
print("="*70)

# ============================================================================
# PART 1: LOAD DATA & TEXT FEATURES
# ============================================================================
print("\n1. Loading data...")
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

def extract_ipq(text):
    if pd.isna(text):
        return 1
    text = str(text).lower()
    if 'pack of 12' in text: return 12
    if 'pack of 6' in text: return 6
    if 'pack of 4' in text: return 4
    patterns = [r'pack of (\d+)', r'(\d+)\s*pack', r'(\d+)\s*count']
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            qty = int(match.group(1))
            if 1 <= qty <= 100:
                return qty
    return 1

print("\n2. Engineering text features...")
for df in [train, test]:
    df['ipq'] = df['catalog_content'].apply(extract_ipq)
    df['ipq_log'] = np.log1p(df['ipq'])
    df['ipq_sqrt'] = np.sqrt(df['ipq'])
    df['ipq_squared'] = df['ipq'] ** 2
    df['text_len'] = df['catalog_content'].fillna('').str.len()
    df['word_count'] = df['catalog_content'].fillna('').str.split().str.len()
    df['has_organic'] = df['catalog_content'].fillna('').str.lower().str.contains('organic').astype(int)
    df['has_premium'] = df['catalog_content'].fillna('').str.lower().str.contains('premium|deluxe|gourmet').astype(int)
    df['ipq_x_len'] = df['ipq'] * df['text_len'] / 1000
    df['ipq_x_word'] = df['ipq'] * df['word_count']

# TF-IDF
print("\n3. Creating TF-IDF features...")
tfidf = TfidfVectorizer(max_features=200, min_df=5, max_df=0.9, ngram_range=(1, 2), sublinear_tf=True)
all_text = pd.concat([train['catalog_content'].fillna(''), test['catalog_content'].fillna('')])
tfidf.fit(all_text)
train_tfidf = tfidf.transform(train['catalog_content'].fillna('')).toarray()
test_tfidf = tfidf.transform(test['catalog_content'].fillna('')).toarray()

# ============================================================================
# PART 2: LOAD ADVANCED IMAGE FEATURES
# ============================================================================
print("\n4. Loading advanced image features...")

try:
    # Try to load numpy files (faster)
    train_img = np.load('features/train_image_features_advanced.npy')
    test_img = np.load('features/test_image_features_advanced.npy')
    print(f"  ✅ Loaded from .npy files")
except:
    # Fallback to CSV
    try:
        train_img = pd.read_csv('features/train_image_features_advanced.csv').values
        test_img = pd.read_csv('features/test_image_features_advanced.csv').values
        print(f"  ✅ Loaded from .csv files")
    except:
        print(f"  ❌ Advanced features not found!")
        print(f"  Run: python advanced_image_feature_engineering.py first")
        exit(1)

print(f"  Shape: {train_img.shape}")

# ============================================================================
# PART 3: COMBINE ALL FEATURES
# ============================================================================
print("\n5. Combining features...")

basic_features = ['ipq', 'ipq_log', 'ipq_sqrt', 'ipq_squared', 'text_len', 
                  'word_count', 'has_organic', 'has_premium', 'ipq_x_len', 'ipq_x_word']

X_train = np.hstack([
    train[basic_features].values,
    train_tfidf,
    train_img
])

X_test = np.hstack([
    test[basic_features].values,
    test_tfidf,
    test_img
])

print(f"  Final shape: {X_train.shape}")
print(f"    - Basic: {len(basic_features)}")
print(f"    - TF-IDF: {train_tfidf.shape[1]}")
print(f"    - Images: {train_img.shape[1]}")
print(f"    - TOTAL: {X_train.shape[1]}")

y = train['price'].values
y_log = np.log1p(y)

# ============================================================================
# PART 4: TRAIN ENSEMBLE MODELS
# ============================================================================
print("\n6. Training ensemble models...")

n_folds = 10  # More folds for better generalization
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# ========== LightGBM ==========
print("\n  Training LightGBM...")
lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 31,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

lgb_oof = np.zeros(len(X_train))
lgb_test = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_log[tr_idx], y_log[val_idx]
    
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=3000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    lgb_oof[val_idx] = model.predict(X_val)
    lgb_test += model.predict(X_test) / n_folds
    
    if fold == 0:  # Print only first fold
        print(f"    Fold 1 SMAPE: {smape(np.expm1(y_val), np.expm1(lgb_oof[val_idx])):.2f}%")

lgb_cv = smape(y, np.expm1(lgb_oof))
print(f"  ✅ LightGBM CV: {lgb_cv:.2f}%")

# ========== XGBoost ==========
print("\n  Training XGBoost...")
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'eta': 0.01,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42,
    'verbosity': 0
}

xgb_oof = np.zeros(len(X_train))
xgb_test = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_log[tr_idx], y_log[val_idx]
    
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=3000,
        evals=[(dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    
    xgb_oof[val_idx] = model.predict(dval)
    xgb_test += model.predict(dtest) / n_folds

xgb_cv = smape(y, np.expm1(xgb_oof))
print(f"  ✅ XGBoost CV: {xgb_cv:.2f}%")

# ========== CatBoost ==========
print("\n  Training CatBoost...")
cat_oof = np.zeros(len(X_train))
cat_test = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_log[tr_idx], y_log[val_idx]
    
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.01,
        depth=8,
        random_seed=42,
        verbose=0
    )
    
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100)
    
    cat_oof[val_idx] = model.predict(X_val)
    cat_test += model.predict(X_test) / n_folds

cat_cv = smape(y, np.expm1(cat_oof))
print(f"  ✅ CatBoost CV: {cat_cv:.2f}%")

# ============================================================================
# PART 5: ENSEMBLE OPTIMIZATION
# ============================================================================
print("\n7. Optimizing ensemble weights...")

best_smape = float('inf')
best_weights = None

for w1 in np.arange(0.3, 0.7, 0.05):
    for w2 in np.arange(0.1, 0.5, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.1 or w3 > 0.5:
            continue
        
        ensemble_oof = w1 * lgb_oof + w2 * xgb_oof + w3 * cat_oof
        ensemble_smape = smape(y, np.expm1(ensemble_oof))
        
        if ensemble_smape < best_smape:
            best_smape = ensemble_smape
            best_weights = (w1, w2, w3)

w1, w2, w3 = best_weights
final_test = w1 * lgb_test + w2 * xgb_test + w3 * cat_test

print(f"  Best weights: LGB={w1:.2f}, XGB={w2:.2f}, CAT={w3:.2f}")

# ============================================================================
# PART 6: RESULTS
# ============================================================================
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"\nIndividual Models:")
print(f"  LightGBM:  {lgb_cv:.2f}%")
print(f"  XGBoost:   {xgb_cv:.2f}%")
print(f"  CatBoost:  {cat_cv:.2f}%")
print(f"\n🏆 Ensemble:  {best_smape:.2f}%")

if best_smape < 12:
    print("\n🎉🎉🎉 AMAZING! SMAPE < 12%!")
    print("TOP-TIER PERFORMANCE!")
elif best_smape < 14:
    print("\n🎊 EXCELLENT! SMAPE < 14%!")
    print("Very competitive!")
elif best_smape < 16:
    print("\n✅ GREAT! SMAPE < 16%!")
    print("Strong performance!")
else:
    print("\n👍 GOOD! Room for hyperparameter tuning")

# Create submission
final_preds = np.expm1(final_test)
final_preds = np.maximum(final_preds, 0.01)

submission = pd.DataFrame({
    'sample_id': test['sample_id'],
    'price': final_preds
})

submission.to_csv('output/test_out_advanced.csv', index=False)

print(f"\n✅ Submission: output/test_out_advanced.csv")
print(f"   Predictions: {len(submission):,}")
print(f"   Price range: ${submission['price'].min():.2f} - ${submission['price'].max():,.2f}")

# Save OOF
ensemble_oof = w1 * lgb_oof + w2 * xgb_oof + w3 * cat_oof
oof_df = pd.DataFrame({
    'sample_id': train['sample_id'],
    'actual': y,
    'predicted': np.expm1(ensemble_oof),
    'error': np.abs(y - np.expm1(ensemble_oof))
})
oof_df.to_csv('output/oof_advanced.csv', index=False)

print("\n" + "="*70)
print(f"FINAL CV SMAPE: {best_smape:.2f}%")
print("="*70)