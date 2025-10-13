"""
STEP 3: Download Images (OPTIONAL)
Can be skipped if you want to use text features only
Execution time: ~2-4 hours (run overnight recommended)
Adds 2-3% SMAPE improvement
"""

import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("STEP 3: IMAGE DOWNLOAD (OPTIONAL)")
print("="*70)

# Create image directories
Path('images/train').mkdir(parents=True, exist_ok=True)
Path('images/test').mkdir(parents=True, exist_ok=True)

# Load data
print("\nLoading data...")
train = pd.read_csv('../dataset/train.csv')
test = pd.read_csv('../dataset/test.csv')

def download_image(url, save_path, timeout=15, retries=3):
    """Download single image with retries"""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                return True
        except Exception:
            if attempt == retries - 1:
                return False
            time.sleep(2)
    return False

def download_batch(df, image_dir, dataset_name, batch_size=500, sleep_between_batches=3):
    """Download images in batches"""
    
    success = 0
    fail = 0
    skip = 0
    
    print(f"\n{dataset_name}: Downloading {len(df)} images...")
    print(f"Saving to: {image_dir}")
    print(f"Batch size: {batch_size}")
    print("-"*70)
    
    # Check existing images
    existing = set()
    for img_path in Path(image_dir).glob('*.jpg'):
        existing.add(img_path.stem)
    
    if existing:
        print(f"Found {len(existing)} existing images, will skip them")
    
    # Download in batches
    for i in tqdm(range(0, len(df), batch_size), desc=f"{dataset_name} batches"):
        batch = df.iloc[i:i+batch_size]
        batch_success = 0
        batch_fail = 0
        batch_skip = 0
        
        for _, row in batch.iterrows():
            sample_id = str(row['sample_id'])
            save_path = Path(image_dir) / f"{sample_id}.jpg"
            
            # Skip if already exists
            if sample_id in existing or save_path.exists():
                skip += 1
                batch_skip += 1
                continue
            
            # Download
            if download_image(row['image_link'], save_path):
                success += 1
                batch_success += 1
            else:
                fail += 1
                batch_fail += 1
        
        # Print batch stats
        print(f"Batch {i//batch_size + 1}: Success={batch_success}, Fail={batch_fail}, Skip={batch_skip}")
        
        # Sleep between batches to avoid throttling
        if i + batch_size < len(df):
            time.sleep(sleep_between_batches)
    
    return success, fail, skip

# User confirmation
print("\n" + "="*70)
print("IMAGE DOWNLOAD INFORMATION")
print("="*70)
print(f"Total images to download: {len(train) + len(test):,}")
print(f"Estimated time: 2-4 hours")
print(f"Bandwidth usage: ~5-10 GB")
print("\nBenefits:")
print("  ✓ Adds 2-3% SMAPE improvement")
print("  ✓ Captures visual product information")
print("\nNote:")
print("  • Can be run overnight")
print("  • Can be skipped entirely (use text features only)")
print("  • Partial downloads are OK (will improve what's available)")

response = input("\nProceed with download? (yes/no): ").strip().lower()

if response != 'yes':
    print("\n⏭️  Skipping image download")
    print("You can still train models with text features only")
    print("Next step: Run step4_modeling.py")
    exit(0)

# Download train images
print("\n" + "="*70)
print("DOWNLOADING TRAIN IMAGES")
print("="*70)
train_success, train_fail, train_skip = download_batch(
    train, 'images/train', 'TRAIN', batch_size=500, sleep_between_batches=3
)

# Download test images
print("\n" + "="*70)
print("DOWNLOADING TEST IMAGES")
print("="*70)
test_success, test_fail, test_skip = download_batch(
    test, 'images/test', 'TEST', batch_size=500, sleep_between_batches=3
)

# Summary
print("\n" + "="*70)
print("STEP 3 COMPLETE!")
print("="*70)

print("\nTRAIN SET:")
print(f"  ✅ Success: {train_success:,} ({train_success/len(train)*100:.1f}%)")
print(f"  ❌ Failed:  {train_fail:,} ({train_fail/len(train)*100:.1f}%)")
print(f"  ⏭️  Skipped: {train_skip:,} ({train_skip/len(train)*100:.1f}%)")

print("\nTEST SET:")
print(f"  ✅ Success: {test_success:,} ({test_success/len(test)*100:.1f}%)")
print(f"  ❌ Failed:  {test_fail:,} ({test_fail/len(test)*100:.1f}%)")
print(f"  ⏭️  Skipped: {test_skip:,} ({test_skip/len(test)*100:.1f}%)")

print("\nTOTAL:")
total_success = train_success + test_success
total_fail = train_fail + test_fail
total_images = len(train) + len(test)
print(f"  ✅ Success: {total_success:,} ({total_success/total_images*100:.1f}%)")
print(f"  ❌ Failed:  {total_fail:,} ({total_fail/total_images*100:.1f}%)")

if total_success / total_images > 0.95:
    print("\n🎉 Excellent! >95% download success rate")
elif total_success / total_images > 0.85:
    print("\n✅ Good! >85% download success rate")
else:
    print("\n⚠️  Warning: Low success rate. Try running again with:")
    print("   - Longer timeout (timeout=30)")
    print("   - More retries (retries=5)")
    print("   - Longer sleep between batches (sleep_between_batches=10)")

print("\nNext step:")
if total_success > 0:
    print("  Run step3b_extract_image_features.py")
else:
    print("  Skip to step4_modeling.py (text features only)")

# Save download log
log_df = pd.DataFrame([{
    'train_success': train_success,
    'train_fail': train_fail,
    'test_success': test_success,
    'test_fail': test_fail,
    'total_success': total_success,
    'success_rate': total_success/total_images*100
}])
log_df.to_csv('output/step3_download_log.csv', index=False)
print("\n✅ Download log saved to: output/step3_download_log.csv")