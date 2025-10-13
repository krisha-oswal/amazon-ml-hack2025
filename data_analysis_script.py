"""
STEP 1: Data Analysis and Understanding
Run this first to understand your dataset
Execution time: ~5-10 minutes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("="*70)
print("STEP 1: DATA ANALYSIS")
print("="*70)

# Create output directory
Path('output').mkdir(exist_ok=True)

# Load data
print("\n1. Loading data...")
try:
    train = pd.read_csv('../dataset/train.csv')
    test = pd.read_csv('../dataset/test.csv')
    print(f"✅ Train: {train.shape}")
    print(f"✅ Test: {test.shape}")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("Make sure train.csv and test.csv are in dataset/ folder")
    exit(1)

# Basic info
print("\n2. Dataset Info")
print("-"*70)
print(f"Training samples: {len(train):,}")
print(f"Test samples: {len(test):,}")
print(f"Features: {train.columns.tolist()}")

# Missing values
print("\n3. Missing Values")
print("-"*70)
print("Train:")
print(train.isnull().sum())
print("\nTest:")
print(test.isnull().sum())

# Price statistics
print("\n4. PRICE STATISTICS (Target Variable)")
print("-"*70)
print(train['price'].describe())
print(f"\nPrice range: ${train['price'].min():.2f} - ${train['price'].max():,.2f}")
print(f"Median: ${train['price'].median():.2f}")

# Outlier detection
q1, q2, q3 = train['price'].quantile([0.25, 0.50, 0.75])
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr
outliers = train['price'] > upper_fence

print(f"\nQ1: ${q1:.2f}")
print(f"Q2 (Median): ${q2:.2f}")
print(f"Q3: ${q3:.2f}")
print(f"IQR: ${iqr:.2f}")
print(f"Upper fence: ${upper_fence:.2f}")
print(f"Outliers (>fence): {outliers.sum()} ({outliers.sum()/len(train)*100:.2f}%)")

# Sample products
print("\n5. Sample Products")
print("-"*70)
for i in [0, 1000, 5000]:
    print(f"\nSample {i}:")
    print(f"Price: ${train.iloc[i]['price']:.2f}")
    print(f"Content: {train.iloc[i]['catalog_content'][:200]}...")

# Visualizations
print("\n6. Creating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Price distribution
axes[0, 0].hist(train['price'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Price Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Price ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(train['price'].median(), color='red', linestyle='--', label='Median')
axes[0, 0].legend()

# Log price distribution
axes[0, 1].hist(np.log1p(train['price']), bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].set_title('Log(Price+1) Distribution', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Log(Price+1)')
axes[0, 1].set_ylabel('Frequency')

# Boxplot
axes[1, 0].boxplot(train['price'], vert=True)
axes[1, 0].set_title('Price Boxplot', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Price ($)')

# Price by percentile
percentiles = np.arange(0, 101, 5)
price_percentiles = np.percentile(train['price'], percentiles)
axes[1, 1].plot(percentiles, price_percentiles, marker='o', linewidth=2)
axes[1, 1].set_title('Price by Percentile', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Percentile')
axes[1, 1].set_ylabel('Price ($)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/step1_price_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Saved to: output/step1_price_analysis.png")

# Summary statistics to file
summary = {
    'total_samples': len(train),
    'price_min': train['price'].min(),
    'price_max': train['price'].max(),
    'price_mean': train['price'].mean(),
    'price_median': train['price'].median(),
    'price_std': train['price'].std(),
    'outliers_count': outliers.sum(),
    'outliers_pct': outliers.sum()/len(train)*100
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv('output/step1_summary.csv', index=False)
print("✅ Saved to: output/step1_summary.csv")

print("\n" + "="*70)
print("STEP 1 COMPLETE!")
print("="*70)
print("\nKey Findings:")
print(f"✓ Total training samples: {len(train):,}")
print(f"✓ Price range: ${train['price'].min():.2f} - ${train['price'].max():,.2f}")
print(f"✓ Median price: ${train['price'].median():.2f}")
print(f"✓ Outliers detected: {outliers.sum()} ({outliers.sum()/len(train)*100:.1f}%)")
print("\nRecommendation:")
print("✓ USE LOG TRANSFORMATION for target variable")
print("✓ Handle outliers carefully in modeling")
print("\nNext step: Run step2_feature_engineering.py")