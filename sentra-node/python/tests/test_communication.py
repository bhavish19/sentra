"""
Unit tests for Communication Protocol and Reconstruction
"""

import pytest
from ml_training.secret_sharing import Share, ShamirSecretSharing
from ml_training.secure_comm import (
    SecureChannel,
    SecureMPCNetwork,
    MessageType,
    create_mpc_network
)
from ml_training.reconstruction import (
    SecureReconstruction,
    MPCReconstructionManager,
    create_reconstruction_manager
)


class TestSecureChannel:
    """Tests for SecureChannel"""
    
    def test_channel_creation(self):
        """Test creating a secure channel"""
        # Note: Actual network channels require sockets
        # This tests the structure
        channel = SecureChannel(
            node_id=1,
            port=8001,
            use_tls=False
        )
        assert channel.node_id == 1
        assert channel.port == 8001
    
    def test_message_types(self):
        """Test message type enumeration"""
        assert MessageType.SHARE_EXCHANGE is not None
        assert MessageType.RECONSTRUCTION_REQUEST is not None
        assert MessageType.RECONSTRUCTION_RESPONSE is not None
        assert MessageType.HEARTBEAT is not None


class TestSecureMPCNetwork:
    """Tests for SecureMPCNetwork"""
    
    def test_network_creation(self):
        """Test creating an MPC network"""
        node_configs = {
            1: {'host': 'localhost', 'port': 8001},
            2: {'host': 'localhost', 'port': 8002}
        }
        
        network = SecureMPCNetwork(node_id=1, node_configs=node_configs)
        assert network.node_id == 1
        assert len(network.node_configs) == 2
    
    def test_create_network_helper(self):
        """Test network creation helper function"""
        node_configs = {
            1: {'host': 'localhost', 'port': 8001},
            2: {'host': 'localhost', 'port': 8002}
        }
        
        network = create_mpc_network(1, node_configs, port=8001)
        assert network is not None
        assert network.node_id == 1


class TestSecureReconstruction:
    """Tests for SecureReconstruction"""
    
    def test_reconstruction_creation(self, n_nodes, threshold):
        """Test creating reconstruction manager"""
        node_configs = {
            i: {'host': 'localhost', 'port': 8000 + i}
            for i in range(1, n_nodes + 1)
        }
        network = SecureMPCNetwork(node_id=1, node_configs=node_configs)
        recon = SecureReconstruction(network, threshold)
        assert recon.t == threshold
    
    def test_reconstruct_from_shares(self, n_nodes, threshold, field_size):
        """Test reconstructing secret from shares"""
        node_configs = {
            i: {'host': 'localhost', 'port': 8000 + i}
            for i in range(1, n_nodes + 1)
        }
        network = SecureMPCNetwork(node_id=1, node_configs=node_configs)
        recon = SecureReconstruction(network, threshold)
        shamir = ShamirSecretSharing(field_size=field_size)
        
        secret = 42
        shares = shamir.share(secret, n_nodes, threshold)
        
        # Reconstruct using reconstruction manager (with context)
        reconstructed = recon.reconstruct_value(shares, context="test")
        assert reconstructed == secret
    
    def test_reconstruct_insufficient_shares(self, n_nodes, threshold, field_size):
        """Test that reconstruction fails with insufficient shares"""
        node_configs = {
            i: {'host': 'localhost', 'port': 8000 + i}
            for i in range(1, n_nodes + 1)
        }
        network = SecureMPCNetwork(node_id=1, node_configs=node_configs)
        recon = SecureReconstruction(network, threshold)
        shamir = ShamirSecretSharing(field_size=field_size)
        
        secret = 42
        shares = shamir.share(secret, n_nodes, threshold)
        
        # Try with only 1 share (need at least threshold+1)
        if threshold >= 1:
            # Should raise RuntimeError when trying to reconstruct with insufficient shares
            import pytest
            with pytest.raises(RuntimeError, match="Insufficient shares"):
                recon.reconstruct_value([shares[0]], context="test", timeout=0.1)


class TestMPCReconstructionManager:
    """Tests for MPCReconstructionManager"""
    
    def test_manager_creation(self, n_nodes, threshold):
        """Test creating reconstruction manager"""
        # Create mock network (simplified)
        node_configs = {
            i: {'host': 'localhost', 'port': 8000 + i}
            for i in range(1, n_nodes + 1)
        }
        network = SecureMPCNetwork(node_id=1, node_configs=node_configs)
        
        manager = MPCReconstructionManager(network, threshold)
        assert manager.t == threshold
    
    def test_reconstruct_with_context(self, n_nodes, threshold, field_size):
        """Test reconstruction with context caching"""
        node_configs = {
            i: {'host': 'localhost', 'port': 8000 + i}
            for i in range(1, n_nodes + 1)
        }
        network = SecureMPCNetwork(node_id=1, node_configs=node_configs)
        manager = MPCReconstructionManager(network, threshold)
        shamir = ShamirSecretSharing(field_size=field_size)
        
        secret = 42
        shares = shamir.share(secret, n_nodes, threshold)
        
        # Reconstruct with context
        reconstructed = manager.get_reconstructed_value(
            shares, context="test_context"
        )
        assert reconstructed == secret
    
    def test_create_manager_helper(self, n_nodes, threshold):
        """Test manager creation helper function"""
        node_configs = {
            i: {'host': 'localhost', 'port': 8000 + i}
            for i in range(1, n_nodes + 1)
        }
        network = SecureMPCNetwork(node_id=1, node_configs=node_configs)
        
        manager = create_reconstruction_manager(network, t=threshold)
        assert manager is not None
        assert manager.t == threshold

