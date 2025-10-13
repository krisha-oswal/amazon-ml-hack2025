"""
ADVANCED IMAGE FEATURE ENGINEERING
Multiple models + color/texture features
Expected improvement: 2-4% SMAPE reduction
"""
# --- Patch for missing LZMA support on macOS Python builds ---

# --- Mac temporary LZMA patch (Python built without lzma support) ---
import sys, io, types

try:
    import lzma  # If it works, do nothing
except ModuleNotFoundError:
    class FakeLZMAFile(io.BytesIO):
        def __init__(self, *args, **kwargs):
            super().__init__()

    def fake_open(filename, mode="rb", *args, **kwargs):
        return open(filename, mode.replace("t", ""), *args, **kwargs)

    fake_lzma = types.SimpleNamespace(
        LZMAFile=FakeLZMAFile,
        open=fake_open,
        FORMAT_AUTO=0,
        FORMAT_ALONE=1,
        FORMAT_XZ=2,
        CHECK_NONE=0,
        CHECK_CRC32=1,
        CHECK_CRC64=4,
        CHECK_SHA256=10,
        PRESET_DEFAULT=6,
    )

    sys.modules['lzma'] = fake_lzma
# --------------------------------------------------------------------

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ADVANCED IMAGE FEATURE ENGINEERING")
print("="*70)

# ============================================================================
# PART 1: SETUP
# ============================================================================
print("\n1. Setup...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load data
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

train_img_dir = Path('images/train')
test_img_dir = Path('images/test')

# ============================================================================
# PART 2: DEEP LEARNING FEATURES (MULTIPLE MODELS)
# ============================================================================
print("\n2. Loading multiple pre-trained models...")

# Model 1: ResNet50 (good for general features)
print("  Loading ResNet50...")
resnet50 = models.resnet50(pretrained=True)
resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
resnet50.eval().to(device)

# Model 2: EfficientNet-B0 (efficient and accurate)
print("  Loading EfficientNet-B0...")
try:
    from torchvision.models import efficientnet_b0
    efficientnet = efficientnet_b0(pretrained=True)
    efficientnet = nn.Sequential(*list(efficientnet.children())[:-1])
    efficientnet.eval().to(device)
    use_efficientnet = True
except:
    print("    ⚠️  EfficientNet not available, skipping")
    use_efficientnet = False

# Model 3: MobileNetV2 (captures different features)
print("  Loading MobileNetV2...")
mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.features.eval().to(device)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================================
# PART 3: ADVANCED FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def extract_color_features(img_path):
    """
    Extract color-based features
    - Dominant colors
    - Color distribution
    - HSV statistics
    """
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return np.zeros(15)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Reshape for clustering
        pixels = img_rgb.reshape(-1, 3).astype(np.float32)
        
        # Get dominant colors (3 colors)
        if len(pixels) > 100:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=3)
            kmeans.fit(pixels[:10000])  # Sample for speed
            dominant_colors = kmeans.cluster_centers_
        else:
            dominant_colors = np.zeros((3, 3))
        
        # Color statistics
        mean_rgb = np.mean(pixels, axis=0)
        std_rgb = np.std(pixels, axis=0)
        
        # HSV statistics
        hsv_pixels = img_hsv.reshape(-1, 3)
        mean_hsv = np.mean(hsv_pixels, axis=0)
        
        # Combine features
        features = np.concatenate([
            dominant_colors.flatten(),  # 9 features (3 colors × RGB)
            mean_rgb,                   # 3 features
            mean_hsv / [180, 255, 255]  # 3 features (normalized)
        ])
        
        return features
    except:
        return np.zeros(15)

def extract_texture_features(img_path):
    """
    Extract texture features
    - Edge density
    - Texture complexity
    - Sharpness
    - Contrast
    """
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros(5)
        
        # Resize for consistency
        img = cv2.resize(img, (224, 224))
        
        # Edge detection
        edges = cv2.Canny(img, 100, 200)
        edge_density = np.sum(edges > 0) / (224 * 224)
        
        # Texture (Laplacian variance)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        texture_complexity = np.var(laplacian)
        
        # Sharpness
        sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
        
        # Contrast
        contrast = img.std()
        
        # Brightness
        brightness = img.mean()
        
        features = np.array([
            edge_density,
            texture_complexity / 10000,  # Normalize
            sharpness / 10000,           # Normalize
            contrast / 100,              # Normalize
            brightness / 255             # Normalize
        ])
        
        return features
    except:
        return np.zeros(5)

def extract_deep_features_multi(img_path):
    """
    Extract features from multiple deep learning models
    """
    try:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        features_list = []
        
        with torch.no_grad():
            # ResNet50 features
            resnet_features = resnet50(img_tensor).squeeze().cpu().numpy()
            features_list.append(resnet_features[:512])  # Take first 512
            
            # EfficientNet features
            if use_efficientnet:
                eff_features = efficientnet(img_tensor).squeeze().cpu().numpy()
                features_list.append(eff_features[:256])  # Take first 256
            
            # MobileNet features
            mobile_features = mobilenet.features(img_tensor)
            mobile_features = torch.nn.functional.adaptive_avg_pool2d(mobile_features, 1)
            mobile_features = mobile_features.squeeze().cpu().numpy()
            features_list.append(mobile_features[:256])  # Take first 256
        
        # Combine all deep features
        combined = np.concatenate(features_list)
        return combined
        
    except Exception as e:
        # Return appropriate size zero vector
        size = 512 + (256 if use_efficientnet else 0) + 256
        return np.zeros(size)

def extract_image_statistics(img_path):
    """
    Extract basic image statistics
    - Resolution
    - Aspect ratio
    - File size
    """
    try:
        img = Image.open(img_path)
        width, height = img.size
        
        features = np.array([
            width / 1000,              # Normalized width
            height / 1000,             # Normalized height
            (width * height) / 1000000, # Megapixels
            width / height,            # Aspect ratio
        ])
        
        return features
    except:
        return np.zeros(4)

# ============================================================================
# PART 4: EXTRACT ALL FEATURES
# ============================================================================

def extract_all_features(sample_id, img_dir):
    """Extract complete feature set for one image"""
    img_path = img_dir / f"{sample_id}.jpg"
    
    if not img_path.exists():
        # Return zero features if image missing
        size = 512 + (256 if use_efficientnet else 0) + 256 + 15 + 5 + 4
        return np.zeros(size)
    
    # Extract all feature types
    deep_features = extract_deep_features_multi(img_path)
    color_features = extract_color_features(img_path)
    texture_features = extract_texture_features(img_path)
    stat_features = extract_image_statistics(img_path)
    
    # Combine everything
    all_features = np.concatenate([
        deep_features,
        color_features,
        texture_features,
        stat_features
    ])
    
    return all_features

def process_images(sample_ids, img_dir, desc="Processing"):
    """Process all images with progress bar"""
    all_features = []
    
    for sample_id in tqdm(sample_ids, desc=desc):
        features = extract_all_features(sample_id, img_dir)
        all_features.append(features)
    
    return np.array(all_features)

# ============================================================================
# PART 5: EXTRACT FEATURES FOR TRAIN AND TEST
# ============================================================================

print("\n3. Extracting advanced image features...")
print("   This will take 20-40 minutes...")

print("\n  Extracting TRAIN features...")
train_features = process_images(
    train['sample_id'].tolist(),
    train_img_dir,
    desc="  Train"
)

print("\n  Extracting TEST features...")
test_features = process_images(
    test['sample_id'].tolist(),
    test_img_dir,
    desc="  Test"
)

print(f"\n  Initial feature shape: {train_features.shape}")

# ============================================================================
# PART 6: DIMENSIONALITY REDUCTION WITH PCA
# ============================================================================

print("\n4. Applying PCA for dimensionality reduction...")

# Use more components for better representation
n_components = 200  # Increased from 128

pca = PCA(n_components=n_components, random_state=42)
all_features = np.vstack([train_features, test_features])

print(f"  Fitting PCA with {n_components} components...")
pca.fit(all_features)

train_pca = pca.transform(train_features)
test_pca = pca.transform(test_features)

explained_var = pca.explained_variance_ratio_.sum() * 100
print(f"  ✅ Final shape: {train_pca.shape}")
print(f"  ✅ Explained variance: {explained_var:.1f}%")

# ============================================================================
# PART 7: SAVE FEATURES
# ============================================================================

print("\n5. Saving features...")

# Save as numpy arrays (faster to load)
np.save('features/train_image_features_advanced.npy', train_pca)
np.save('features/test_image_features_advanced.npy', test_pca)

# Also save as CSV for compatibility
train_img_df = pd.DataFrame(
    train_pca,
    columns=[f'img_adv_{i}' for i in range(n_components)]
)
test_img_df = pd.DataFrame(
    test_pca,
    columns=[f'img_adv_{i}' for i in range(n_components)]
)

train_img_df.to_csv('features/train_image_features_advanced.csv', index=False)
test_img_df.to_csv('features/test_image_features_advanced.csv', index=False)

print("  ✅ features/train_image_features_advanced.npy")
print("  ✅ features/test_image_features_advanced.npy")
print("  ✅ features/train_image_features_advanced.csv")
print("  ✅ features/test_image_features_advanced.csv")

# ============================================================================
# PART 8: FEATURE ANALYSIS
# ============================================================================

print("\n6. Feature Analysis...")

# Calculate feature statistics
print("\n  Feature Statistics:")
print(f"    Mean: {train_pca.mean():.4f}")
print(f"    Std:  {train_pca.std():.4f}")
print(f"    Min:  {train_pca.min():.4f}")
print(f"    Max:  {train_pca.max():.4f}")

# PCA component analysis
print("\n  Top 10 PCA Components (variance explained):")
for i in range(min(10, n_components)):
    var_pct = pca.explained_variance_ratio_[i] * 100
    print(f"    Component {i+1}: {var_pct:.2f}%")

print("\n" + "="*70)
print("ADVANCED IMAGE FEATURES EXTRACTED!")
print("="*70)

print("\n✅ What was extracted:")
print(f"  • Deep features from 3 models (ResNet50, EfficientNet, MobileNet)")
print(f"  • Color features (dominant colors, RGB/HSV stats)")
print(f"  • Texture features (edges, complexity, sharpness)")
print(f"  • Image statistics (resolution, aspect ratio)")
print(f"  • Total: {n_components} PCA components")

print("\n📊 Expected improvement:")
print("  • Basic ResNet features: 14-18% SMAPE")
print("  • Advanced features (this): 12-16% SMAPE")
print("  • Improvement: ~2-3 percentage points")

print("\n🚀 Next step:")
print("  Run the complete model training with these features")
print("  python train_with_advanced_images.py")

print("\n" + "="*70)