"""
Key-Value Store with Versioning
Enclave-backed KVS with rollback protection and freshness guarantees
"""

from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from enum import Enum


@dataclass
class KVSValue:
    """Value stored in KVS with version"""
    data: Any
    version: int
    timestamp: float


class KVSNode:
    """
    Single KVS node
    In production, this would run inside an enclave
    """
    
    def __init__(self, node_id: int):
        """
        Initialize KVS node
        Args:
            node_id: Unique node identifier
        """
        self.node_id = node_id
        self.store: Dict[str, KVSValue] = {}
    
    def write(self, key: str, data: Any, version: int) -> bool:
        """
        Write data with version
        Only accepts writes with version > current version
        Args:
            key: Key to write
            data: Data to store
            version: Version number (must be > current version)
        Returns:
            True if write succeeded, False otherwise
        """
        import time
        
        if key in self.store:
            current_version = self.store[key].version
            if version <= current_version:
                return False  # Version not fresh
        
        self.store[key] = KVSValue(
            data=data,
            version=version,
            timestamp=time.time()
        )
        return True
    
    def read(self, key: str, min_version: int = 0) -> Optional[KVSValue]:
        """
        Read data with minimum version requirement
        Args:
            key: Key to read
            min_version: Minimum version required
        Returns:
            KVSValue if found and version >= min_version, None otherwise
        """
        if key not in self.store:
            return None
        
        value = self.store[key]
        if value.version < min_version:
            return None  # Version too old
        
        return value


class KVSCluster:
    """
    Cluster of KVS nodes with quorum-based operations
    """
    
    def __init__(self, node_ids: List[int]):
        """
        Initialize KVS cluster
        Args:
            node_ids: List of node IDs in the cluster
        """
        self.nodes: Dict[int, KVSNode] = {node_id: KVSNode(node_id) for node_id in node_ids}
        self.n_nodes = len(node_ids)
    
    def write_with_quorum(self, key: str, data: Any, version: int, quorum_size: Optional[int] = None) -> bool:
        """
        Write to cluster with quorum consensus
        Args:
            key: Key to write
            data: Data to write
            version: Version number
            quorum_size: Required quorum size (default: majority)
        Returns:
            True if quorum achieved, False otherwise
        """
        if quorum_size is None:
            quorum_size = (self.n_nodes // 2) + 1
        
        # Write to all nodes
        successes = 0
        for node in self.nodes.values():
            if node.write(key, data, version):
                successes += 1
        
        return successes >= quorum_size
    
    def read_with_min_version(self, key: str, min_version: int = 0) -> Optional[KVSValue]:
        """
        Read from cluster with minimum version
        Returns value from first node that has version >= min_version
        Args:
            key: Key to read
            min_version: Minimum version required
        Returns:
            KVSValue if found, None otherwise
        """
        for node in self.nodes.values():
            value = node.read(key, min_version)
            if value is not None:
                return value
        return None
    
    def get_latest_version(self, key: str) -> int:
        """
        Get latest version for a key across all nodes
        Args:
            key: Key to check
        Returns:
            Latest version number, or 0 if key doesn't exist
        """
        max_version = 0
        for node in self.nodes.values():
            if key in node.store:
                max_version = max(max_version, node.store[key].version)
        return max_version


