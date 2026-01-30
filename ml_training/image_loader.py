"""
Image Loading and Preprocessing for Kather Texture Dataset
Loads images from Kather_texture_2016_image_tiles_5000 folder
"""

import os
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image
import glob


def augment_image(img_array: np.ndarray) -> List[np.ndarray]:
    """
    Apply simple data augmentation to an image
    
    Args:
        img_array: Image array [H x W x C] or [H x W]
    
    Returns:
        List of augmented images (original + augmented versions)
    """
    augmented = [img_array.copy()]  # Original image
    
    # Horizontal flip
    if len(img_array.shape) == 3:
        augmented.append(np.fliplr(img_array).copy())
    else:
        augmented.append(np.fliplr(img_array).copy())
    
    # Small rotation (90 degrees)
    augmented.append(np.rot90(img_array, k=1).copy())
    
    return augmented


def load_kather_dataset(data_dir: str = "Kather_texture_2016_image_tiles_5000",
                        image_size: Tuple[int, int] = (32, 32),
                        grayscale: bool = True,
                        normalize: bool = True,
                        augment: bool = False) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load Kather texture dataset
    
    Args:
        data_dir: Path to dataset folder
        image_size: Target image size (width, height). Default (32, 32) for LeNet-5
        grayscale: Convert to grayscale (True) or keep RGB (False)
        normalize: Normalize pixel values to [0, 1] range
    
    Returns:
        Tuple of (images, labels)
        - images: List of numpy arrays (each image is H x W or H x W x C)
        - labels: List of integer labels (0-7 for 8 classes)
    """
    images = []
    labels = []
    
    # Class mapping
    class_folders = [
        "01_TUMOR",      # 0
        "02_STROMA",     # 1
        "03_COMPLEX",    # 2
        "04_LYMPHO",     # 3
        "05_DEBRIS",     # 4
        "06_MUCOSA",     # 5
        "07_ADIPOSE",    # 6
        "08_EMPTY"       # 7
    ]
    
    print(f"Loading Kather texture dataset from: {data_dir}")
    print(f"Image size: {image_size}, Grayscale: {grayscale}, Normalize: {normalize}")
    
    for class_idx, class_folder in enumerate(class_folders):
        class_path = os.path.join(data_dir, class_folder)
        
        if not os.path.exists(class_path):
            print(f"Warning: Class folder not found: {class_path}")
            continue
        
        # Find all .tif files
        image_files = glob.glob(os.path.join(class_path, "*.tif"))
        image_files.sort()  # Sort for reproducibility
        
        print(f"  Loading {len(image_files)} images from {class_folder} (class {class_idx})")
        
        for img_file in image_files:
            try:
                # Load image
                img = Image.open(img_file)
                
                # Convert to grayscale if needed
                if grayscale:
                    if img.mode != 'L':
                        img = img.convert('L')
                
                # Resize to target size
                img = img.resize(image_size, Image.Resampling.LANCZOS)
                
                # Convert to numpy array
                img_array = np.array(img, dtype=np.float32)
                
                # Normalize to [0, 1] if requested
                if normalize:
                    img_array = img_array / 255.0
                
                # Add channel dimension if grayscale (for consistency)
                if grayscale and len(img_array.shape) == 2:
                    img_array = img_array[:, :, np.newaxis]  # H x W x 1
                
                if augment:
                    # Apply augmentation: creates 3 versions (original + flip + rotate)
                    augmented_imgs = augment_image(img_array)
                    images.extend(augmented_imgs)
                    labels.extend([class_idx] * len(augmented_imgs))
                else:
                    images.append(img_array)
                    labels.append(class_idx)
                
            except Exception as e:
                print(f"Warning: Failed to load {img_file}: {e}")
                continue
    
    print(f"\nLoaded {len(images)} images total")
    print(f"Class distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cls, count in zip(unique, counts):
        class_name = class_folders[cls]
        print(f"  Class {cls} ({class_name}): {count} images")
    
    return images, labels


def prepare_dataset_for_training(images: List[np.ndarray], labels: List[int],
                                 flatten: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Prepare dataset for training
    
    Args:
        images: List of image arrays
        labels: List of integer labels
        flatten: If True, flatten images to 1D vectors (for FC layers)
                 If False, keep 2D/3D shape (for CNN layers)
    
    Returns:
        Tuple of (processed_images, one_hot_labels)
    """
    processed_images = []
    one_hot_labels = []
    
    num_classes = len(set(labels))
    
    for img, label in zip(images, labels):
        # Flatten if needed (for fully connected layers)
        if flatten:
            img_flat = img.flatten()
        else:
            img_flat = img
        
        processed_images.append(img_flat)
        
        # One-hot encode label
        one_hot = np.zeros(num_classes, dtype=np.float32)
        one_hot[label] = 1.0
        one_hot_labels.append(one_hot)
    
    return processed_images, one_hot_labels


def split_dataset(images: List[np.ndarray], labels: List[int],
                  train_ratio: float = 0.8, val_ratio: float = 0.1,
                  test_ratio: float = 0.1, shuffle: bool = True,
                  seed: Optional[int] = None) -> Tuple:
    """
    Split dataset into train/val/test sets
    
    Args:
        images: List of images
        labels: List of labels
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set (should sum to 1.0)
        shuffle: Whether to shuffle before splitting
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_images, train_labels, val_images, val_labels, test_images, test_labels)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    if seed is not None:
        np.random.seed(seed)
    
    # Shuffle
    indices = np.arange(len(images))
    if shuffle:
        np.random.shuffle(indices)
    
    images_shuffled = [images[i] for i in indices]
    labels_shuffled = [labels[i] for i in indices]
    
    # Calculate split points
    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split
    train_images = images_shuffled[:n_train]
    train_labels = labels_shuffled[:n_train]
    
    val_images = images_shuffled[n_train:n_train + n_val]
    val_labels = labels_shuffled[n_train:n_train + n_val]
    
    test_images = images_shuffled[n_train + n_val:]
    test_labels = labels_shuffled[n_train + n_val:]
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_images)} samples")
    print(f"  Val: {len(val_images)} samples")
    print(f"  Test: {len(test_images)} samples")
    
    return train_images, train_labels, val_images, val_labels, test_images, test_labels
