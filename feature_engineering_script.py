"""
STEP 2: Feature Engineering
Extract all features from text data
Execution time: ~30-45 minutes
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("STEP 2: FEATURE ENGINEERING")
print("="*70)

# Create features directory
Path('features').mkdir(exist_ok=True)

# Load data
print("\nLoading data...")
train = pd.read_csv('/Users/kriii/Desktop/amazon-ml/student_resource/dataset/train.csv')
test = pd.read_csv('/Users/kriii/Desktop/amazon-ml/student_resource/dataset/test.csv')

print(f"✅ Train: {train.shape}")
print(f"✅ Test: {test.shape}")

def extract_ipq(text):
    """Extract Item Pack Quantity - CRITICAL FEATURE"""
    if pd.isna(text):
        return 1
    
    text = text.lower()
    patterns = [
        r'pack of (\d+)', r'\(pack of (\d+)\)', r'(\d+)\s*per\s*case',
        r'pk[:\s-]*(\d+)', r'(\d+)\s*pack', r'(\d+)\s*count',
        r'(\d+)\s*ct\b', r'set of (\d+)', r'case of (\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            qty = int(match.group(1))
            if 1 <= qty <= 500:
                return qty
    return 1

def extract_unit_size(text):
    """Extract size in ounces"""
    if pd.isna(text):
        return 0
    
    text = text.lower()
    
    # Ounces
    oz_match = re.search(r'(\d+\.?\d*)\s*oz\b', text)
    if oz_match:
        return float(oz_match.group(1))
    
    # Pounds to oz
    lb_match = re.search(r'(\d+\.?\d*)\s*lb\b', text)
    if lb_match:
        return float(lb_match.group(1)) * 16
    
    # Grams to oz
    g_match = re.search(r'(\d+\.?\d*)\s*g\b', text)
    if g_match:
        return float(g_match.group(1)) * 0.035274
    
    return 0

def extract_brand_tier(text):
    """Classify brand tier"""
    if pd.isna(text):
        return 1
    
    text = text.lower()
    
    premium = ['goya', 'member\'s mark', 'salerno', 'judee\'s', 'organic', 'vineco']
    mid = ['bear creek', 'kedem', 'la victoria']
    
    for brand in premium:
        if brand in text:
            return 3
    for brand in mid:
        if brand in text:
            return 2
    return 1

def count_bullet_points(text):
    """Count bullet points"""
    if pd.isna(text):
        return 0
    return len(re.findall(r'Bullet Point \d+:', text))

def extract_value_field(text):
    """Extract Value: field"""
    if pd.isna(text):
        return 0
    match = re.search(r'Value:\s*(\d+\.?\d*)', text)
    return float(match.group(1)) if match else 0

# Feature extraction
print("\n" + "-"*70)
print("EXTRACTING FEATURES")
print("-"*70)

for df, name in [(train, 'train'), (test, 'test')]:
    print(f"\nProcessing {name} set...")
    
    # 1. IPQ (Most Important!)
    print("  1/12 - IPQ...")
    df['ipq'] = df['catalog_content'].apply(extract_ipq)
    df['ipq_log'] = np.log1p(df['ipq'])
    df['ipq_sqrt'] = np.sqrt(df['ipq'])
    
    # 2. Unit size
    print("  2/12 - Unit size...")
    df['unit_size_oz'] = df['catalog_content'].apply(extract_unit_size)
    df['size_log'] = np.log1p(df['unit_size_oz'])
    
    # 3. Brand tier
    print("  3/12 - Brand tier...")
    df['brand_tier'] = df['catalog_content'].apply(extract_brand_tier)
    
    # 4. Bullet points
    print("  4/12 - Bullet points...")
    df['num_bullets'] = df['catalog_content'].apply(count_bullet_points)
    
    # 5. Value field
    print("  5/12 - Value field...")
    df['value_field'] = df['catalog_content'].apply(extract_value_field)
    df['value_log'] = np.log1p(df['value_field'])
    
    # 6. Text statistics
    print("  6/12 - Text statistics...")
    df['text_length'] = df['catalog_content'].fillna('').str.len()
    df['word_count'] = df['catalog_content'].fillna('').str.split().str.len()
    df['num_digits'] = df['catalog_content'].fillna('').apply(lambda x: sum(c.isdigit() for c in str(x)))
    
    # 7. Quality signals
    print("  7/12 - Quality signals...")
    df['has_organic'] = df['catalog_content'].fillna('').str.lower().str.contains('organic').astype(int)
    df['has_natural'] = df['catalog_content'].fillna('').str.lower().str.contains('natural').astype(int)
    df['has_glutenfree'] = df['catalog_content'].fillna('').str.lower().str.contains('gluten-free|gluten free').astype(int)
    df['has_usa'] = df['catalog_content'].fillna('').str.lower().str.contains('made in usa|made in the usa').astype(int)
    
    # 8. Premium keywords
    print("  8/12 - Premium keywords...")
    premium_kw = ['premium', 'gourmet', 'artisan', 'chef', 'original', 'authentic', 'deluxe']
    df['premium_count'] = df['catalog_content'].fillna('').str.lower().apply(
        lambda x: sum(1 for kw in premium_kw if kw in x)
    )
    
    # 9. Categories
    print("  9/12 - Categories...")
    df['cat_food'] = df['catalog_content'].fillna('').str.lower().str.contains('sauce|soup|cereal|cookie|snack').astype(int)
    df['cat_beverage'] = df['catalog_content'].fillna('').str.lower().str.contains('wine|drink|juice').astype(int)
    df['cat_condiment'] = df['catalog_content'].fillna('').str.lower().str.contains('seasoning|powder|spice').astype(int)
    
    # 10. Unit types
    print("  10/12 - Unit types...")
    df['unit_ounce'] = df['catalog_content'].fillna('').str.contains('Unit: Ounce|Unit: Oz', case=False).astype(int)
    df['unit_count'] = df['catalog_content'].fillna('').str.contains('Unit: Count', case=False).astype(int)
    
    # 11. Interaction features (CRITICAL!)
    print("  11/12 - Interaction features...")
    df['ipq_x_brand'] = df['ipq'] * df['brand_tier']
    df['ipq_x_premium'] = df['ipq'] * df['premium_count']
    df['size_per_item'] = df['unit_size_oz'] / (df['ipq'] + 1)
    df['total_volume'] = df['unit_size_oz'] * df['ipq']
    
    # 12. Price signal
    print("  12/12 - Price signal...")
    df['price_signal'] = (
        df['ipq'] * 0.3 +
        df['brand_tier'] * 0.2 +
        df['unit_size_oz'] * 0.001 +
        df['premium_count'] * 0.2
    )

# TF-IDF features
print("\n" + "-"*70)
print("CREATING TF-IDF FEATURES")
print("-"*70)

tfidf = TfidfVectorizer(
    max_features=150,
    min_df=5,
    max_df=0.9,
    ngram_range=(1, 2),
    sublinear_tf=True
)

all_text = pd.concat([
    train['catalog_content'].fillna(''),
    test['catalog_content'].fillna('')
])

print("Fitting TF-IDF...")
tfidf.fit(all_text)

print("Transforming train...")
train_tfidf = tfidf.transform(train['catalog_content'].fillna(''))
train_tfidf_df = pd.DataFrame(
    train_tfidf.toarray(),
    columns=[f'tfidf_{i}' for i in range(150)]
)

print("Transforming test...")
test_tfidf = tfidf.transform(test['catalog_content'].fillna(''))
test_tfidf_df = pd.DataFrame(
    test_tfidf.toarray(),
    columns=[f'tfidf_{i}' for i in range(150)]
)

# Combine features
print("\n" + "-"*70)
print("COMBINING FEATURES")
print("-"*70)

feature_cols = [
    'ipq', 'ipq_log', 'ipq_sqrt', 'unit_size_oz', 'size_log', 'brand_tier',
    'num_bullets', 'value_field', 'value_log', 'text_length', 'word_count',
    'num_digits', 'has_organic', 'has_natural', 'has_glutenfree', 'has_usa',
    'premium_count', 'cat_food', 'cat_beverage', 'cat_condiment',
    'unit_ounce', 'unit_count', 'ipq_x_brand', 'ipq_x_premium',
    'size_per_item', 'total_volume', 'price_signal'
]

train_final = pd.concat([
    train[['sample_id', 'price']],
    train[feature_cols],
    train_tfidf_df
], axis=1)

test_final = pd.concat([
    test[['sample_id']],
    test[feature_cols],
    test_tfidf_df
], axis=1)

# Save
print("\nSaving features...")
train_final.to_csv('features/train_features_text.csv', index=False)
test_final.to_csv('features/test_features_text.csv', index=False)

print("\n" + "="*70)
print("STEP 2 COMPLETE!")
print("="*70)
print(f"\n✅ Train features: {train_final.shape}")
print(f"✅ Test features: {test_final.shape}")
print(f"✅ Total features: {len(feature_cols) + 150}")
print(f"\nFiles saved:")
print(f"  - features/train_features_text.csv")
print(f"  - features/test_features_text.csv")

# Feature importance preview
print("\n" + "-"*70)
print("FEATURE STATISTICS")
print("-"*70)
print(f"IPQ distribution: min={train['ipq'].min()}, max={train['ipq'].max()}, mean={train['ipq'].mean():.2f}")
print(f"IPQ value counts (top 10):")
print(train['ipq'].value_counts().head(10))
print(f"\nBrand tier distribution:")
print(train['brand_tier'].value_counts())

print("\nNext step: Run step3_download_images.py (optional, can be skipped)")
print("Or skip to step4_modeling.py if you don't want image features")