"""
STEP 3B: Extract Image Features (OPTIONAL)
Only run if you downloaded images in Step 3
Execution time: ~1-2 hours
"""


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("STEP 3B: IMAGE FEATURE EXTRACTION")
print("="*70)

# Check if images exist
train_images = list(Path('src/images/train').glob('*.jpg'))
test_images = list(Path('src/images/test').glob('*.jpg'))

if not train_images:
    print("\n❌ No images found in images/train/")
    print("Run step3_download_images.py first, or skip to step4_modeling.py")
    exit(1)

print(f"\n✅ Found {len(train_images)} train images")
print(f"✅ Found {len(test_images)} test images")

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Using device: {device}")

# Load ResNet50
print("\nLoading ResNet50...")
model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # Remove FC layer
model.eval().to(device)
print("✅ Model loaded!")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def extract_features(sample_ids, image_dir, batch_size=64):
    """Extract features from images"""
    features = []
    
    print(f"\nExtracting features from {len(sample_ids)} images...")
    print(f"Batch size: {batch_size}")
    
    for i in tqdm(range(0, len(sample_ids), batch_size)):
        batch_ids = sample_ids[i:i+batch_size]
        batch_features = []
        
        for sample_id in batch_ids:
            img_path = Path(image_dir) / f"{sample_id}.jpg"
            
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    feature = model(img_tensor).squeeze().cpu().numpy()
                
                batch_features.append(feature)
            except:
                # Zero vector for missing/corrupt images
                batch_features.append(np.zeros(2048))
        
        features.extend(batch_features)
    
    return np.array(features)

# Load data
print("\nLoading data...")
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

# Extract features
print("\n" + "-"*70)
print("EXTRACTING TRAIN IMAGE FEATURES")
print("-"*70)
train_img_features = extract_features(
    train['sample_id'].tolist(),
    'images/train',
    batch_size=64
)
print(f"✅ Train features shape: {train_img_features.shape}")

print("\n" + "-"*70)
print("EXTRACTING TEST IMAGE FEATURES")
print("-"*70)
test_img_features = extract_features(
    test['sample_id'].tolist(),
    'images/test',
    batch_size=64
)
print(f"✅ Test features shape: {test_img_features.shape}")

# Dimensionality reduction with PCA
print("\n" + "-"*70)
print("APPLYING PCA DIMENSIONALITY REDUCTION")
print("-"*70)

n_components = 128
print(f"Reducing from 2048 to {n_components} dimensions...")

pca = PCA(n_components=n_components, random_state=42)

# Fit on combined data
all_features = np.vstack([train_img_features, test_img_features])
pca.fit(all_features)

# Transform
train_img_pca = pca.transform(train_img_features)
test_img_pca = pca.transform(test_img_features)

explained_var = pca.explained_variance_ratio_.sum()
print(f"✅ Explained variance: {explained_var:.4f} ({explained_var*100:.2f}%)")
print(f"✅ Train PCA shape: {train_img_pca.shape}")
print(f"✅ Test PCA shape: {test_img_pca.shape}")

# Convert to DataFrame
train_img_df = pd.DataFrame(
    train_img_pca,
    columns=[f'img_{i}' for i in range(n_components)]
)
test_img_df = pd.DataFrame(
    test_img_pca,
    columns=[f'img_{i}' for i in range(n_components)]
)

# Save image features
print("\nSaving image features...")
train_img_df.to_csv('features/train_image_features.csv', index=False)
test_img_df.to_csv('features/test_image_features.csv', index=False)
print("✅ Saved to: features/train_image_features.csv")
print("✅ Saved to: features/test_image_features.csv")

# Combine with text features
print("\n" + "-"*70)
print("COMBINING TEXT + IMAGE FEATURES")
print("-"*70)

print("Loading text features...")
train_text = pd.read_csv('features/train_features_text.csv')
test_text = pd.read_csv('features/test_features_text.csv')

print("Combining...")
train_combined = pd.concat([train_text, train_img_df], axis=1)
test_combined = pd.concat([test_text, test_img_df], axis=1)

print("Saving combined features...")
train_combined.to_csv('features/train_features_final.csv', index=False)
test_combined.to_csv('features/test_features_final.csv', index=False)

print("\n" + "="*70)
print("STEP 3B COMPLETE!")
print("="*70)
print(f"\n✅ Train combined: {train_combined.shape}")
print(f"✅ Test combined: {test_combined.shape}")
print(f"✅ Total features: {train_combined.shape[1] - 2}")  # Minus sample_id and price

print("\nFiles saved:")
print("  - features/train_image_features.csv")
print("  - features/test_image_features.csv")
print("  - features/train_features_final.csv")
print("  - features/test_features_final.csv")

print("\nNext step: Run step4_modeling.py")