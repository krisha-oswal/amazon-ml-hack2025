"""
DIAGNOSTIC & FIX SCRIPT
Run this to identify and fix performance issues
"""

import pandas as pd
import numpy as np
import re
import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PERFORMANCE DIAGNOSTIC & FIX")
print("="*70)

# Load data
print("\nLoading data...")
train = pd.read_csv('../dataset/train.csv')

def smape(y_true, y_pred):
    """Calculate SMAPE"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

print("\n" + "="*70)
print("STEP 1: CHECK IPQ EXTRACTION")
print("="*70)

def extract_ipq_improved(text):
    """Improved IPQ extraction"""
    if pd.isna(text):
        return 1
    
    text_str = str(text).lower()
    
    # More comprehensive patterns
    patterns = [
        (r'pack of (\d+)', 1.0),           # "Pack of 6"
        (r'\(pack of (\d+)\)', 1.0),       # "(Pack of 6)"
        (r'(\d+)\s*per\s*case', 1.0),      # "12 per case"
        (r'case of (\d+)', 1.0),           # "case of 12"
        (r'pk[:\s-]*(\d+)', 1.0),          # "PK-6"
        (r'(\d+)\s*pk\b', 1.0),            # "6 pk"
        (r'(\d+)\s*pack', 1.0),            # "6 pack"
        (r'(\d+)\s*count', 1.0),           # "6 count"
        (r'(\d+)\s*ct\b', 1.0),            # "6 ct"
        (r'set of (\d+)', 1.0),            # "set of 6"
        (r'(\d+)\s*piece', 1.0),           # "6 piece"
        (r'quantity[:\s]*(\d+)', 1.0),     # "Quantity: 6"
        (r'(\d+)\s*units?', 1.0),          # "6 units"
        (r'(\d+)\s*bottles?', 1.0),        # "6 bottles"
        (r'(\d+)\s*cans?', 1.0),           # "6 cans"
        (r'(\d+)\s*jars?', 1.0),           # "6 jars"
        (r'(\d+)\s*boxes?', 1.0),          # "6 boxes"
    ]
    
    for pattern, confidence in patterns:
        matches = re.findall(pattern, text_str)
        if matches:
            # Take the first match
            qty = int(matches[0])
            # Sanity check
            if 1 <= qty <= 500:
                return qty
    
    return 1  # Default

# Test IPQ extraction
print("\nTesting IPQ extraction on sample data:")
sample_indices = [0, 100, 1000, 5000, 10000]
for idx in sample_indices:
    text = train.iloc[idx]['catalog_content']
    ipq = extract_ipq_improved(text)
    price = train.iloc[idx]['price']
    print(f"\nSample {idx}:")
    print(f"  IPQ: {ipq}")
    print(f"  Price: ${price:.2f}")
    print(f"  Text: {text[:150]}...")

# Apply to all data
train['ipq'] = train['catalog_content'].apply(extract_ipq_improved)

print("\n" + "-"*70)
print("IPQ Distribution:")
print(train['ipq'].value_counts().head(20))
print(f"\nIPQ Statistics:")
print(f"  Min: {train['ipq'].min()}")
print(f"  Max: {train['ipq'].max()}")
print(f"  Mean: {train['ipq'].mean():.2f}")
print(f"  Median: {train['ipq'].median()}")

# Check correlation with price
print(f"\nIPQ vs Price Correlation: {train['ipq'].corr(train['price']):.4f}")
print("  (Should be > 0.5 for good performance)")

print("\n" + "="*70)
print("STEP 2: FEATURE ENGINEERING")
print("="*70)

def create_optimized_features(df):
    """Create optimized feature set"""
    
    print("Creating features...")
    
    # IPQ features (MOST IMPORTANT)
    df['ipq_log'] = np.log1p(df['ipq'])
    df['ipq_sqrt'] = np.sqrt(df['ipq'])
    df['ipq_squared'] = df['ipq'] ** 2
    
    # Extract more information
    def extract_oz(text):
        if pd.isna(text): return 0
        match = re.search(r'(\d+\.?\d*)\s*oz', str(text).lower())
        return float(match.group(1)) if match else 0
    
    def extract_value(text):
        if pd.isna(text): return 0
        match = re.search(r'value[:\s]*(\d+\.?\d*)', str(text).lower())
        return float(match.group(1)) if match else 0
    
    def count_bullets(text):
        if pd.isna(text): return 0
        return len(re.findall(r'bullet point \d+:', str(text).lower()))
    
    df['unit_oz'] = df['catalog_content'].apply(extract_oz)
    df['value_field'] = df['catalog_content'].apply(extract_value)
    df['num_bullets'] = df['catalog_content'].apply(count_bullets)
    
    # Text features
    df['text_length'] = df['catalog_content'].fillna('').str.len()
    df['word_count'] = df['catalog_content'].fillna('').str.split().str.len()
    df['has_organic'] = df['catalog_content'].fillna('').str.lower().str.contains('organic').astype(int)
    df['has_premium'] = df['catalog_content'].fillna('').str.lower().str.contains('premium|deluxe|gourmet').astype(int)
    
    # CRITICAL: Interaction features
    df['ipq_x_oz'] = df['ipq'] * df['unit_oz']
    df['ipq_x_value'] = df['ipq'] * df['value_field']
    df['ipq_x_bullets'] = df['ipq'] * df['num_bullets']
    df['oz_per_item'] = df['unit_oz'] / (df['ipq'] + 1)
    
    # Log transformations
    df['unit_oz_log'] = np.log1p(df['unit_oz'])
    df['value_log'] = np.log1p(df['value_field'])
    
    return df

train = create_optimized_features(train)

print("\n" + "="*70)
print("STEP 3: MODEL TRAINING WITH OPTIMIZED SETTINGS")
print("="*70)

feature_cols = [
    'ipq', 'ipq_log', 'ipq_sqrt', 'ipq_squared',
    'unit_oz', 'unit_oz_log', 'value_field', 'value_log',
    'num_bullets', 'text_length', 'word_count',
    'has_organic', 'has_premium',
    'ipq_x_oz', 'ipq_x_value', 'ipq_x_bullets', 'oz_per_item'
]

X = train[feature_cols].values
y = train['price'].values

# CRITICAL: Use log transformation
y_log = np.log1p(y)

print(f"\nFeatures: {len(feature_cols)}")
print(f"Samples: {len(X):,}")
print(f"Price range: ${y.min():.2f} - ${y.max():,.2f}")

# Optimized LightGBM parameters
params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

# 5-fold CV for quick test
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

print(f"\nTraining with {n_folds}-fold CV...")

oof_predictions = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}/{n_folds}")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_log[train_idx], y_log[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(100)
        ]
    )
    
    oof_predictions[val_idx] = model.predict(X_val)
    
    # Calculate SMAPE for this fold
    fold_smape = smape(np.expm1(y_val), np.expm1(oof_predictions[val_idx]))
    print(f"  Fold {fold + 1} SMAPE: {fold_smape:.2f}%")

# Overall CV score
cv_smape = smape(y, np.expm1(oof_predictions))

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"\n✅ CV SMAPE: {cv_smape:.2f}%")

if cv_smape < 15:
    print("🎉 EXCELLENT! Much improved!")
elif cv_smape < 25:
    print("✅ GOOD! Significant improvement!")
elif cv_smape < 35:
    print("👍 BETTER! Getting there...")
else:
    print("⚠️  Still needs work. Check diagnostics below.")

print("\n" + "="*70)
print("FEATURE IMPORTANCE")
print("="*70)

importance = model.feature_importance()
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': importance
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features:")
print(importance_df.head(10).to_string(index=False))

print("\n" + "="*70)
print("DIAGNOSTICS")
print("="*70)

# Check predictions
oof_prices = np.expm1(oof_predictions)
errors = np.abs(y - oof_prices)
pct_errors = errors / y * 100

print(f"\nPrediction Quality:")
print(f"  Mean Absolute Error: ${errors.mean():.2f}")
print(f"  Median Absolute Error: ${np.median(errors):.2f}")
print(f"  Mean % Error: {pct_errors.mean():.2f}%")

# Worst predictions
print("\nWorst 5 Predictions:")
worst_idx = errors.argsort()[-5:][::-1]
for idx in worst_idx:
    print(f"\n  Sample {idx}:")
    print(f"    Actual: ${y[idx]:.2f}")
    print(f"    Predicted: ${oof_prices[idx]:.2f}")
    print(f"    Error: ${errors[idx]:.2f}")
    print(f"    IPQ: {train.iloc[idx]['ipq']}")
    print(f"    Text: {train.iloc[idx]['catalog_content'][:100]}...")

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

if cv_smape > 30:
    print("\n⚠️  SMAPE still high. Try:")
    print("  1. Check if IPQ extraction is working correctly")
    print("  2. Verify log transformation is applied")
    print("  3. Add more interaction features")
    print("  4. Use TF-IDF features")
elif cv_smape > 15:
    print("\n✅ Good progress! To improve further:")
    print("  1. Add TF-IDF features (150 dimensions)")
    print("  2. Use ensemble of LightGBM + XGBoost + CatBoost")
    print("  3. Increase to 10-fold CV")
    print("  4. Add image features")
else:
    print("\n🎉 Excellent! You're on track for <10% with:")
    print("  1. Image features (+2-3% improvement)")
    print("  2. Ensemble models (+1-2% improvement)")
    print("  3. Hyperparameter tuning (+0.5-1% improvement)")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("\nIf SMAPE improved significantly:")
print("  1. Run full pipeline: python step2_features.py")
print("  2. Then: python step4_modeling.py")
print("\nIf SMAPE still high (>30%):")
print("  1. Check your data files are correct")
print("  2. Verify IPQ extraction patterns match your data")
print("  3. Look at the 'Worst Predictions' section above")

# Save improved model
print("\nSaving diagnostic results...")
diag_results = pd.DataFrame({
    'sample_id': train['sample_id'],
    'actual_price': y,
    'predicted_price': oof_prices,
    'error': errors,
    'pct_error': pct_errors,
    'ipq': train['ipq']
})
diag_results.to_csv('output/diagnostic_results.csv', index=False)
print("✅ Saved to: output/diagnostic_results.csv")

print("\n" + "="*70)