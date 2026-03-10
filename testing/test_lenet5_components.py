"""
Test script for LeNet-5 components
Tests individual components to verify they work correctly
"""

import sys
import os
import argparse
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml_training.image_loader import load_kather_dataset, prepare_dataset_for_training
from ml_training.secret_sharing import Share, ShamirSecretSharing
from ml_training.secure_convolution import SecureConvolution
from ml_training.secure_pooling import SecurePooling
from ml_training.lenet5 import LeNet5
from ml_training.beaver_triples import BeaverTripleGenerator, BeaverTriplePool, SecureMultiplier
from ml_training.secure_comparison import SecureComparator
from ml_training.secure_division import SecureDivider


def test_image_loading():
    """Test 1: Image Loading"""
    print("\n[TEST] Image Loading")
    print("-" * 60)
    
    try:
        images, labels = load_kather_dataset(
            data_dir="Kather_texture_2016_image_tiles_5000",
            image_size=(32, 32),
            grayscale=True,
            normalize=True
        )
        
        assert len(images) > 0, "No images loaded"
        assert len(images) == len(labels), "Images and labels mismatch"
        assert len(set(labels)) == 8, f"Expected 8 classes, got {len(set(labels))}"
        
        # Check image shape
        img_shape = images[0].shape
        assert img_shape == (32, 32, 1) or img_shape == (32, 32), f"Unexpected image shape: {img_shape}"
        
        print(f"✓ Loaded {len(images)} images")
        print(f"✓ Image shape: {img_shape}")
        print(f"✓ Number of classes: {len(set(labels))}")
        print("✓ Image loading test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Image loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_share_conversion():
    """Test 2: Share Conversion"""
    print("\n[TEST] Share Conversion")
    print("-" * 60)
    
    try:
        # Load one image
        images, labels = load_kather_dataset(
            data_dir="Kather_texture_2016_image_tiles_5000",
            image_size=(32, 32),
            grayscale=True,
            normalize=True
        )
        
        image = images[0]
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        
        # Convert to shares
        n_nodes = 5
        t = 1
        node_id = 1
        field_size = 2**31 - 1
        shamir = ShamirSecretSharing(field_size)
        
        H, W, C = image.shape
        image_shares = []
        
        for h in range(H):
            row_shares = []
            for w in range(W):
                channel_shares = []
                for c in range(C):
                    pixel_value = float(image[h, w, c])
                    pixel_int = int(pixel_value * 1000000) % field_size
                    shares = shamir.share(pixel_int, n_nodes, t)
                    node_share = next(s for s in shares if s.node_id == node_id)
                    channel_shares.append(node_share)
                row_shares.append(channel_shares)
            image_shares.append(row_shares)
        
        assert len(image_shares) == H, f"Height mismatch: {len(image_shares)} != {H}"
        assert len(image_shares[0]) == W, f"Width mismatch: {len(image_shares[0])} != {W}"
        assert len(image_shares[0][0]) == C, f"Channel mismatch: {len(image_shares[0][0])} != {C}"
        
        print(f"✓ Image shape: {image.shape}")
        print(f"✓ Share shape: ({len(image_shares)}, {len(image_shares[0])}, {len(image_shares[0][0])})")
        print("✓ Share conversion test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Share conversion test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_convolution():
    """Test 3: Convolution Forward Pass"""
    print("\n[TEST] Convolution Forward Pass")
    print("-" * 60)
    
    try:
        # Setup
        n_nodes = 5
        t = 1
        node_id = 1
        field_size = 2**31 - 1
        
        triple_gen = BeaverTripleGenerator(field_size)
        triple_pool = BeaverTriplePool(triple_gen, initial_size=100)
        multiplier = SecureMultiplier(triple_pool, n_nodes, t, field_size)
        conv_op = SecureConvolution(multiplier, field_size)
        
        # Create dummy input shares (32x32x1)
        shamir = ShamirSecretSharing(field_size)
        input_shares = []
        for h in range(32):
            row = []
            for w in range(32):
                channel = []
                val_int = 1000000 % field_size
                shares = shamir.share(val_int, n_nodes, t)
                node_share = next(s for s in shares if s.node_id == node_id)
                channel.append(node_share)
                row.append(channel)
            input_shares.append(row)
        
        # Create dummy kernel (5x5x1x6)
        kernel_shares = []
        for k_h in range(5):
            kernel_row = []
            for k_w in range(5):
                kernel_ch_in = []
                for c_in in range(1):
                    kernel_ch_out = []
                    for c_out in range(6):
                        val_int = 500000 % field_size
                        shares = shamir.share(val_int, n_nodes, t)
                        node_share = next(s for s in shares if s.node_id == node_id)
                        kernel_ch_out.append(node_share)
                    kernel_ch_in.append(kernel_ch_out)
                kernel_row.append(kernel_ch_in)
            kernel_shares.append(kernel_row)
        
        # Forward pass
        output = conv_op.conv2d(
            input_shares, kernel_shares, stride=1, padding=0,
            node_id=node_id, context="test_conv"
        )
        
        # Check output dimensions: (32-5+1) x (32-5+1) x 6 = 28x28x6
        assert len(output) == 28, f"Output height mismatch: {len(output)} != 28"
        assert len(output[0]) == 28, f"Output width mismatch: {len(output[0])} != 28"
        assert len(output[0][0]) == 6, f"Output channels mismatch: {len(output[0][0])} != 6"
        
        print(f"✓ Input shape: (32, 32, 1)")
        print(f"✓ Kernel shape: (5, 5, 1, 6)")
        print(f"✓ Output shape: ({len(output)}, {len(output[0])}, {len(output[0][0])})")
        print("✓ Convolution test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Convolution test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pooling():
    """Test 4: Pooling Forward Pass"""
    print("\n[TEST] Pooling Forward Pass")
    print("-" * 60)
    
    try:
        # Setup
        n_nodes = 5
        t = 1
        node_id = 1
        field_size = 2**31 - 1
        
        triple_gen = BeaverTripleGenerator(field_size)
        triple_pool = BeaverTriplePool(triple_gen, initial_size=100)
        multiplier = SecureMultiplier(triple_pool, n_nodes, t, field_size)
        divider = SecureDivider(multiplier, field_size)
        pool_op = SecurePooling(divider=divider, field_size=field_size)
        
        # Create dummy input shares (28x28)
        shamir = ShamirSecretSharing(field_size)
        input_shares = []
        for h in range(28):
            row = []
            for w in range(28):
                val_int = 1000000 % field_size
                shares = shamir.share(val_int, n_nodes, t)
                node_share = next(s for s in shares if s.node_id == node_id)
                row.append(node_share)
            input_shares.append(row)
        
        # Forward pass (max pooling, 2x2, stride 2)
        output = pool_op.max_pool2d(
            input_shares, pool_size=(2, 2), stride=2,
            node_id=node_id, context="test_pool"
        )
        
        # Check output dimensions: (28/2) x (28/2) = 14x14
        assert len(output) == 14, f"Output height mismatch: {len(output)} != 14"
        assert len(output[0]) == 14, f"Output width mismatch: {len(output[0])} != 14"
        
        print(f"✓ Input shape: (28, 28)")
        print(f"✓ Pool size: (2, 2)")
        print(f"✓ Output shape: ({len(output)}, {len(output[0])})")
        print("✓ Pooling test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Pooling test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lenet5_forward():
    """Test 5: LeNet-5 Forward Pass"""
    print("\n[TEST] LeNet-5 Forward Pass")
    print("-" * 60)
    
    try:
        # Setup
        n_nodes = 5
        t = 1
        node_id = 1
        field_size = 2**31 - 1
        
        triple_gen = BeaverTripleGenerator(field_size)
        triple_pool = BeaverTriplePool(triple_gen, initial_size=1000)
        multiplier = SecureMultiplier(triple_pool, n_nodes, t, field_size)
        comparator = SecureComparator(multiplier, field_size)
        divider = SecureDivider(multiplier, field_size)
        
        lenet5 = LeNet5(
            n_nodes=n_nodes,
            t=t,
            multiplier=multiplier,
            field_size=field_size,
            comparator=comparator,
            divider=divider
        )
        
        # Initialize weights
        weights = lenet5.initialize_weights(node_id=node_id)
        
        # Create dummy input (32x32x1)
        shamir = ShamirSecretSharing(field_size)
        input_shares = []
        for h in range(32):
            row = []
            for w in range(32):
                channel = []
                val_int = 1000000 % field_size
                shares = shamir.share(val_int, n_nodes, t)
                node_share = next(s for s in shares if s.node_id == node_id)
                channel.append(node_share)
                row.append(channel)
            input_shares.append(row)
        
        # Forward pass
        output, intermediates = lenet5.forward_pass(
            input_shares, weights, node_id=node_id,
            context="test_lenet5", return_intermediates=True
        )
        
        # Check output
        assert len(output) == 8, f"Output size mismatch: {len(output)} != 8"
        assert 'conv1_out' in intermediates, "Missing conv1_out in intermediates"
        assert 'pool1_out' in intermediates, "Missing pool1_out in intermediates"
        assert 'conv2_out' in intermediates, "Missing conv2_out in intermediates"
        assert 'pool2_out' in intermediates, "Missing pool2_out in intermediates"
        
        print("✓ Forward pass completed")
        print(f"✓ Output size: {len(output)} (expected 8)")
        print(f"✓ Intermediates cached: {len(intermediates)} items")
        print("✓ LeNet-5 forward pass test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ LeNet-5 forward pass test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_loop():
    """Test 6: Training Loop"""
    print("\n[TEST] Training Loop")
    print("-" * 60)
    
    try:
        # Load small subset
        images, labels = load_kather_dataset(
            data_dir="Kather_texture_2016_image_tiles_5000",
            image_size=(32, 32),
            grayscale=True,
            normalize=True
        )
        
        # Use only first 10 images for quick test
        test_images = images[:10]
        test_labels = labels[:10]
        
        processed_images, one_hot_labels = prepare_dataset_for_training(
            test_images, test_labels, flatten=False
        )
        
        # Setup
        n_nodes = 5
        t = 1
        node_id = 1
        field_size = 2**31 - 1
        
        triple_gen = BeaverTripleGenerator(field_size)
        triple_pool = BeaverTriplePool(triple_gen, initial_size=1000)
        multiplier = SecureMultiplier(triple_pool, n_nodes, t, field_size)
        comparator = SecureComparator(multiplier, field_size)
        divider = SecureDivider(multiplier, field_size)
        
        lenet5 = LeNet5(
            n_nodes=n_nodes,
            t=t,
            multiplier=multiplier,
            field_size=field_size,
            comparator=comparator,
            divider=divider
        )
        
        weights = lenet5.initialize_weights(node_id=node_id)
        
        # Process one batch
        batch_size = 2
        batch_images = processed_images[:batch_size]
        batch_labels = [one_hot_labels[i] for i in range(batch_size)]
        
        # Convert to shares (simplified)
        shamir = ShamirSecretSharing(field_size)
        batch_image_shares = []
        for img in batch_images:
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
            H, W, C = img.shape
            img_shares = []
            for h in range(H):
                row = []
                for w in range(W):
                    channel = []
                    for c in range(C):
                        val_int = int(img[h, w, c] * 1000000) % field_size
                        shares = shamir.share(val_int, n_nodes, t)
                        node_share = next(s for s in shares if s.node_id == node_id)
                        channel.append(node_share)
                    row.append(channel)
                img_shares.append(row)
            batch_image_shares.append(img_shares)
        
        # Forward pass
        for i, img_shares in enumerate(batch_image_shares):
            output = lenet5.forward_pass(
                img_shares, weights, node_id=node_id,
                context=f"batch_{i}", return_intermediates=False
            )
            assert len(output) == 8, f"Output size mismatch: {len(output)} != 8"
        
        print(f"✓ Processed {len(batch_images)} images")
        print(f"✓ Forward pass completed for all images")
        print("✓ Training loop test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Training loop test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("LeNet-5 Component Tests")
    print("=" * 60)
    
    tests = [
        ("Image Loading", test_image_loading),
        ("Share Conversion", test_share_conversion),
        ("Convolution", test_convolution),
        ("Pooling", test_pooling),
        ("LeNet-5 Forward", test_lenet5_forward),
        ("Training Loop", test_training_loop),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test CRASHED: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! LeNet-5 implementation is working.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check errors above.")
    
    return passed == total


def main():
    parser = argparse.ArgumentParser(description='Test LeNet-5 components')
    parser.add_argument('--test', type=str, default='all',
                       choices=['all', 'image_loading', 'share_conversion', 
                               'convolution', 'pooling', 'lenet5_forward', 'training_loop'],
                       help='Which test to run')
    
    args = parser.parse_args()
    
    if args.test == 'all':
        success = run_all_tests()
        sys.exit(0 if success else 1)
    elif args.test == 'image_loading':
        success = test_image_loading()
    elif args.test == 'share_conversion':
        success = test_share_conversion()
    elif args.test == 'convolution':
        success = test_convolution()
    elif args.test == 'pooling':
        success = test_pooling()
    elif args.test == 'lenet5_forward':
        success = test_lenet5_forward()
    elif args.test == 'training_loop':
        success = test_training_loop()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
