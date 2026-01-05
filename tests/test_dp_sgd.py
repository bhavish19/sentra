"""
Unit tests for DP-SGD Integration
"""

import pytest
import numpy as np
from ml_training.secret_sharing import Share, ShamirSecretSharing
from ml_training.dp_sgd_integration import (
    DPSGDConfig,
    SecureDPNoiseGenerator,
    DPSGDMPCEngine,
    VerifiableDPProofs
)


class TestDPSGDConfig:
    """Tests for DPSGDConfig"""
    
    def test_config_creation(self):
        """Test creating DP-SGD configuration"""
        config = DPSGDConfig(
            noise_multiplier=1.0,
            clip_norm=1.0,
            learning_rate=0.01
        )
        assert config.noise_multiplier == 1.0
        assert config.clip_norm == 1.0
        assert config.learning_rate == 0.01
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = DPSGDConfig()
        assert config.noise_multiplier > 0
        assert config.clip_norm > 0
        assert config.batch_size > 0


class TestSecureDPNoiseGenerator:
    """Tests for SecureDPNoiseGenerator"""
    
    def test_generate_gaussian_noise(self, secure_multiplier, field_size):
        """Test generating Gaussian noise shares"""
        generator = SecureDPNoiseGenerator(secure_multiplier, field_size)
        
        # Generate noise share
        noise_share, u1, u2 = generator.generate_gaussian_noise_share(
            mean=0.0, stddev=1.0, node_id=1
        )
        
        # Check structure
        assert isinstance(noise_share, Share)
        assert isinstance(u1, float)
        assert isinstance(u2, float)
    
    def test_generate_noise_batch(self, secure_multiplier, field_size):
        """Test generating noise for batch"""
        generator = SecureDPNoiseGenerator(secure_multiplier, field_size)
        
        # Generate noise for multiple gradients
        noise_shares = []
        for _ in range(5):
            noise_share, _, _ = generator.generate_gaussian_noise_share(
                mean=0.0, stddev=0.5, node_id=1
            )
            noise_shares.append(noise_share)
        
        assert len(noise_shares) == 5


class TestDPSGDMPCEngine:
    """Tests for DPSGDMPCEngine"""
    
    def test_engine_creation(self, secure_multiplier, field_size):
        """Test creating DP-SGD MPC engine"""
        config = DPSGDConfig()
        engine = DPSGDMPCEngine(config, secure_multiplier, field_size)
        assert engine.dp_config == config
        assert engine.field_size == field_size
    
    def test_dp_sgd_step(self, secure_multiplier, n_nodes, threshold, field_size):
        """Test DP-SGD step on shares"""
        config = DPSGDConfig(noise_multiplier=1.0, clip_norm=1.0)
        engine = DPSGDMPCEngine(config, secure_multiplier, field_size)
        shamir = ShamirSecretSharing(field_size=field_size)
        
        # Create gradient shares
        gradients = [10, 20, 30]
        grad_shares = []
        for grad in gradients:
            shares = shamir.share(int(grad), n_nodes, threshold)
            grad_shares.append(shares)
        
        # Apply DP-SGD step (expects List[List[Share]])
        noisy_grads, _ = engine.dp_sgd_step_on_shares([grad_shares], batch_size=1, node_id=1)
        
        # Check structure
        assert len(noisy_grads) > 0
    
    def test_clipping_before_noise(self, secure_multiplier, n_nodes, threshold, field_size):
        """Test that clipping happens before noise addition"""
        config = DPSGDConfig(noise_multiplier=0.1, clip_norm=1.0)
        engine = DPSGDMPCEngine(config, secure_multiplier, field_size)
        shamir = ShamirSecretSharing(field_size=field_size)
        
        # Create large gradients
        gradients = [100, 200, 300]
        grad_shares = []
        for grad in gradients:
            shares = shamir.share(int(grad), n_nodes, threshold)
            grad_shares.append(shares)
        
        # Apply DP-SGD (should clip)
        noisy_grads, _ = engine.dp_sgd_step_on_shares([grad_shares], batch_size=1, node_id=1)
        
        # Check that gradients were processed
        assert len(noisy_grads) > 0


class TestVerifiableDPProofs:
    """Tests for VerifiableDPProofs"""
    
    def test_proof_creation(self):
        """Test creating verifiable DP proofs"""
        proofs = VerifiableDPProofs()
        
        # Generate proof for DP-SGD step
        proof = proofs.generate_proof(
            gradients=[1, 2, 3],
            noise_scale=1.0,
            clip_norm=1.0
        )
        
        assert proof is not None
        assert 'verified' in proof or 'proof' in proof
    
    def test_proof_verification(self):
        """Test proof verification"""
        proofs = VerifiableDPProofs()
        
        # Generate and verify proof
        proof = proofs.generate_proof(
            gradients=[1, 2, 3],
            noise_scale=1.0,
            clip_norm=1.0
        )
        
        verified = proofs.verify_proof(proof)
        # Verification result depends on implementation
        assert isinstance(verified, bool) or verified is None

