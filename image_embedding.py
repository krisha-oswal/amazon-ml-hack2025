"""
ADVANCED: Image Feature Extraction using Pre-trained Models
Extract deep visual features from product images
Execution time: ~2-3 hours (depending on number of images)
"""
import sys, types

# Create a fake lzma module with a dummy LZMAFile class
class DummyLZMAFile:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("lzma compression not supported on this system")

fake_lzma = types.SimpleNamespace(LZMAFile=DummyLZMAFile)
sys.modules['lzma'] = fake_lzma

import pandas as pd
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ADVANCED IMAGE FEATURE EXTRACTION")
print("="*70)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Create directories
Path('features').mkdir(exist_ok=True)

# ============================================================================
# MULTI-MODEL ENSEMBLE FOR IMAGE EMBEDDINGS
# ============================================================================

class ImageEmbeddingExtractor:
    """Extract embeddings from multiple pre-trained models"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.models = {}
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("\nLoading pre-trained models...")
        
        # 1. ResNet50 (2048 features)
        print("  Loading ResNet50...")
        resnet = models.resnet50(pretrained=True)
        self.models['resnet50'] = torch.nn.Sequential(*list(resnet.children())[:-1])
        
        # 2. EfficientNet-B3 (1536 features) - Better than ResNet
        print("  Loading EfficientNet-B3...")
        efficientnet = models.efficientnet_b3(pretrained=True)
        self.models['efficientnet'] = torch.nn.Sequential(*list(efficientnet.children())[:-1])
        
        # 3. Vision Transformer (768 features) - State-of-the-art
        print("  Loading Vision Transformer...")
        vit = models.vit_b_16(pretrained=True)
        # Remove classification head
        self.models['vit'] = torch.nn.Sequential(*list(vit.children())[:-1])
        
        # Set all models to eval mode
        for model in self.models.values():
            model.eval()
            model.to(device)
        
        print(f"✅ Loaded {len(self.models)} models")
    
    def extract_features(self, img_path):
        """Extract features from all models"""
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transforms(img).unsqueeze(0).to(self.device)
            
            features = []
            with torch.no_grad():
                # ResNet50 features
                resnet_feat = self.models['resnet50'](img_tensor)
                resnet_feat = resnet_feat.squeeze().cpu().numpy()
                features.append(resnet_feat)
                
                # EfficientNet features
                eff_feat = self.models['efficientnet'](img_tensor)
                eff_feat = eff_feat.squeeze().cpu().numpy()
                features.append(eff_feat)
                
                # ViT features
                vit_feat = self.models['vit'](img_tensor)
                vit_feat = vit_feat.squeeze().cpu().numpy()
                features.append(vit_feat)
            
            # Concatenate all features
            return np.concatenate(features)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            # Return zero vector if image fails
            return np.zeros(2048 + 1536 + 768)

# ============================================================================
# EXTRACT EMBEDDINGS
# ============================================================================

print("\n" + "="*70)
print("EXTRACTING IMAGE EMBEDDINGS")
print("="*70)

# Initialize extractor
extractor = ImageEmbeddingExtractor(device=device)

# Load sample IDs
train_df = pd.read_csv('/Users/kriii/Desktop/amazon-ml/student_resource/dataset/train.csv')
test_df = pd.read_csv('/Users/kriii/Desktop/amazon-ml/student_resource/dataset/test.csv')

image_dir = Path('/Users/kriii/Desktop/amazon-ml/student_resource/dataset/images')

def process_dataset(df, name):
    """Process all images in dataset"""
    print(f"\nProcessing {name} set ({len(df)} samples)...")
    
    embeddings = []
    valid_indices = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = row['sample_id']
        img_path = image_dir / f"{sample_id}.jpg"
        
        if img_path.exists():
            emb = extractor.extract_features(img_path)
            embeddings.append(emb)
            valid_indices.append(idx)
        else:
            # Use zero embedding for missing images
            embeddings.append(np.zeros(2048 + 1536 + 768))
            valid_indices.append(idx)
    
    embeddings = np.array(embeddings)
    print(f"✅ Extracted embeddings: {embeddings.shape}")
    
    return embeddings

# Extract embeddings
train_embeddings = process_dataset(train_df, 'train')
test_embeddings = process_dataset(test_df, 'test')

# ============================================================================
# DIMENSIONALITY REDUCTION (Optional but Recommended)
# ============================================================================

print("\n" + "="*70)
print("DIMENSIONALITY REDUCTION")
print("="*70)

from sklearn.decomposition import PCA

# Reduce from 4352 to 512 dimensions (keeps 95%+ variance)
print("\nApplying PCA...")
pca = PCA(n_components=512, random_state=42)

# Fit on combined data
all_embeddings = np.vstack([train_embeddings, test_embeddings])
pca.fit(all_embeddings)

train_embeddings_pca = pca.transform(train_embeddings)
test_embeddings_pca = pca.transform(test_embeddings)

print(f"✅ Reduced to {train_embeddings_pca.shape[1]} dimensions")
print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# ============================================================================
# SAVE EMBEDDINGS
# ============================================================================

print("\n" + "="*70)
print("SAVING EMBEDDINGS")
print("="*70)

# Create DataFrames
train_img_features = pd.DataFrame(
    train_embeddings_pca,
    columns=[f'img_emb_{i}' for i in range(512)]
)
train_img_features.insert(0, 'sample_id', train_df['sample_id'])

test_img_features = pd.DataFrame(
    test_embeddings_pca,
    columns=[f'img_emb_{i}' for i in range(512)]
)
test_img_features.insert(0, 'sample_id', test_df['sample_id'])

# Save
train_img_features.to_csv('features/train_image_embeddings.csv', index=False)
test_img_features.to_csv('features/test_image_embeddings.csv', index=False)

print("\n✅ Saved embeddings:")
print(f"   - features/train_image_embeddings.csv")
print(f"   - features/test_image_embeddings.csv")

# ============================================================================
# COMBINE WITH TEXT FEATURES
# ============================================================================

print("\n" + "="*70)
print("COMBINING WITH TEXT FEATURES")
print("="*70)

# Load text features
train_text = pd.read_csv('features/train_features_text.csv')
test_text = pd.read_csv('features/test_features_text.csv')

# Merge on sample_id
train_combined = train_text.merge(train_img_features, on='sample_id', how='left')
test_combined = test_text.merge(test_img_features, on='sample_id', how='left')

# Fill missing image features with 0
img_cols = [col for col in train_combined.columns if col.startswith('img_emb_')]
train_combined[img_cols] = train_combined[img_cols].fillna(0)
test_combined[img_cols] = test_combined[img_cols].fillna(0)

# Save combined features
train_combined.to_csv('features/train_features_final.csv', index=False)
test_combined.to_csv('features/test_features_final.csv', index=False)

print(f"\n✅ Combined features saved:")
print(f"   Train: {train_combined.shape}")
print(f"   Test: {test_combined.shape}")
print(f"   Total features: {len(train_combined.columns) - 2}")

print("\n" + "="*70)
print("IMAGE EXTRACTION COMPLETE!")
print("="*70)
print("\nFeature breakdown:")
print(f"  Text features: {len(train_text.columns) - 2}")
print(f"  Image embeddings: {len(img_cols)}")
print(f"  Total: {len(train_combined.columns) - 2}")
print("\nNext: Run the advanced modeling script!")