"""
MINIMAL WORKING SOLUTION
This should get you to 20-25% SMAPE minimum
Simple, focused approach that works
"""

import pandas as pd
import numpy as np
import re
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

def smape(y_true, y_pred):
    """SMAPE metric"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

print("="*70)
print("MINIMAL WORKING SOLUTION")
print("="*70)

# Load data
print("\n1. Loading data...")
train = pd.read_csv('../dataset/train.csv')
test = pd.read_csv('../dataset/test.csv')
print(f"   Train: {train.shape}")
print(f"   Test: {test.shape}")

print("\n2. Extracting KEY features...")

def extract_ipq_robust(text):
    """Ultra-robust IPQ extraction"""
    if pd.isna(text):
        return 1
    
    text = str(text).lower()
    
    # Try all common patterns
    patterns = [
        r'pack of (\d+)', r'\(pack of (\d+)\)', r'(\d+)\s*per\s*case',
        r'(\d+)\s*pack', r'(\d+)\s*count', r'case of (\d+)',
        r'set of (\d+)', r'(\d+)\s*ct\b', r'(\d+)\s*pk\b'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            qty = int(match.group(1))
            if 1 <= qty <= 500:
                return qty
    
    return 1

# Extract IPQ for both train and test
train['ipq'] = train['catalog_content'].apply(extract_ipq_robust)
test['ipq'] = test['catalog_content'].apply(extract_ipq_robust)

print(f"   IPQ extracted (train): min={train['ipq'].min()}, max={train['ipq'].max()}, mean={train['ipq'].mean():.1f}")
print(f"   IPQ extracted (test): min={test['ipq'].min()}, max={test['ipq'].max()}, mean={test['ipq'].mean():.1f}")

# Create simple but powerful features
print("\n3. Creating features...")

for df in [train, test]:
    # IPQ transformations
    df['ipq_log'] = np.log1p(df['ipq'])
    df['ipq_sqrt'] = np.sqrt(df['ipq'])
    
    # Text features
    df['text_len'] = df['catalog_content'].fillna('').str.len()
    df['word_count'] = df['catalog_content'].fillna('').str.split().str.len()
    
    # Simple pattern matching
    text_lower = df['catalog_content'].fillna('').str.lower()
    df['has_organic'] = text_lower.str.contains('organic').astype(int)
    df['has_premium'] = text_lower.str.contains('premium|deluxe|gourmet').astype(int)
    df['has_wine'] = text_lower.str.contains('wine').astype(int)
    
    # Interaction
    df['ipq_x_len'] = df['ipq'] * df['text_len'] / 1000

print("   ✅ Basic features created")

# TF-IDF features
print("\n4. Creating TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=100,  # Start with 100
    min_df=5,
    max_df=0.9,
    ngram_range=(1, 2)
)

all_text = pd.concat([
    train['catalog_content'].fillna(''),
    test['catalog_content'].fillna('')
])

tfidf.fit(all_text)
train_tfidf = tfidf.transform(train['catalog_content'].fillna(''))
test_tfidf = tfidf.transform(test['catalog_content'].fillna(''))

print(f"   ✅ TF-IDF features: {train_tfidf.shape[1]}")

# Combine features
base_features = ['ipq', 'ipq_log', 'ipq_sqrt', 'text_len', 'word_count',
                 'has_organic', 'has_premium', 'has_wine', 'ipq_x_len']

X_train = np.hstack([
    train[base_features].values,
    train_tfidf.toarray()
])

X_test = np.hstack([
    test[base_features].values,
    test_tfidf.toarray()
])

y = train['price'].values
y_log = np.log1p(y)  # CRITICAL: Log transform

print(f"\n5. Final feature shape: {X_train.shape}")

# Train model
print("\n6. Training LightGBM...")

params = {
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

# 5-fold CV
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"\n   Fold {fold + 1}/{n_folds}")
    
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_log[tr_idx], y_log[val_idx]
    
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(0)
        ]
    )
    
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / n_folds
    
    fold_smape = smape(np.expm1(y_val), np.expm1(oof_preds[val_idx]))
    print(f"   Fold {fold + 1} SMAPE: {fold_smape:.2f}%")

# Final CV score
cv_smape = smape(y, np.expm1(oof_preds))

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"\n📊 CV SMAPE: {cv_smape:.2f}%")

if cv_smape < 15:
    print("🎉 EXCELLENT! Ready for competition!")
elif cv_smape < 25:
    print("✅ GOOD! Significant improvement!")
elif cv_smape < 35:
    print("👍 BETTER! On the right track!")
else:
    print("⚠️  Check IPQ extraction and data quality")

# Create submission
print("\n7. Creating submission...")

final_test_preds = np.expm1(test_preds)
final_test_preds = np.maximum(final_test_preds, 0.01)  # Ensure positive

submission = pd.DataFrame({
    'sample_id': test['sample_id'],
    'price': final_test_preds
})

submission.to_csv('output/test_out_minimal.csv', index=False)

print(f"\n✅ Submission saved: output/test_out_minimal.csv")
print(f"   Rows: {len(submission):,}")
print(f"   Price range: ${submission['price'].min():.2f} - ${submission['price'].max():,.2f}")
print(f"   Median: ${submission['price'].median():.2f}")

# Save OOF for analysis
oof_df = pd.DataFrame({
    'sample_id': train['sample_id'],
    'actual': y,
    'predicted': np.expm1(oof_preds),
    'error': np.abs(y - np.expm1(oof_preds)),
    'ipq': train['ipq']
})
oof_df.to_csv('output/oof_minimal.csv', index=False)
print(f"✅ OOF predictions saved: output/oof_minimal.csv")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nYour CV SMAPE: {cv_smape:.2f}%")
print(f"Previous SMAPE: 53%")
print(f"Improvement: {53 - cv_smape:.2f} percentage points")

if cv_smape < 53:
    print("\n✅ SUCCESS! Model is working much better!")
else:
    print("\n⚠️  Still need to investigate. Check:")
    print("   1. IPQ values in oof_minimal.csv")
    print("   2. Worst predictions")
    print("   3. Data quality")

print("\n" + "="*70)