"""
Beaver Triple Generation and Secure Multiplication
Implements secure multiplication protocol using pre-computed Beaver triples
"""

from typing import List, Tuple, Optional
from ml_training.secret_sharing import Share, ShamirSecretSharing
from ml_training.reconstruction import MPCReconstructionManager
import secrets
import random
import hashlib
import numpy as np


class BeaverTriple:
    """
    Represents a Beaver triple (a, b, c) where c = a * b
    Each value is secret-shared across nodes
    """
    
    def __init__(self, a: List[Share], b: List[Share], c: List[Share]):
        """
        Initialize Beaver triple
        Args:
            a: Secret-shared value a
            b: Secret-shared value b
            c: Secret-shared value c = a * b
        """
        self.a = a
        self.b = b
        self.c = c
        # Speed up lookups (hot path): avoid linear scans in get_for_node()
        self._a_by_node = {s.node_id: s for s in a}
        self._b_by_node = {s.node_id: s for s in b}
        self._c_by_node = {s.node_id: s for s in c}
    
    def get_for_node(self, node_id: int) -> Tuple[Share, Share, Share]:
        """Get triple shares for a specific node"""
        return self._a_by_node[node_id], self._b_by_node[node_id], self._c_by_node[node_id]


class BeaverTripleGenerator:
    """
    Generates Beaver triples using Shamir Secret Sharing
    """
    
    def __init__(self, field_size: int = 2**31 - 1):
        """
        Initialize triple generator
        Args:
            field_size: Prime field size
        """
        self.field_size = field_size
        self.sss = ShamirSecretSharing(field_size)
    
    def generate_triple(self, n_nodes: int, t: int) -> BeaverTriple:
        """
        Generate a Beaver triple (a, b, c) where c = a * b
        Args:
            n_nodes: Number of nodes
            t: Privacy threshold
        Returns:
            BeaverTriple with secret-shared values
        """
        # Generate random values a and b
        a = random.randint(0, self.field_size - 1)
        b = random.randint(0, self.field_size - 1)
        
        # Compute c = a * b
        c = (a * b) % self.field_size
        
        # Secret-share all three values
        a_shares = self.sss.share(a, n_nodes, t)
        b_shares = self.sss.share(b, n_nodes, t)
        c_shares = self.sss.share(c, n_nodes, t)
        
        return BeaverTriple(a_shares, b_shares, c_shares)


class BeaverTriplePool:
    """
    Pool of pre-generated Beaver triples for efficient secure multiplication
    """
    
    def __init__(self, generator: BeaverTripleGenerator, initial_size: int = 100):
        """
        Initialize triple pool
        Args:
            generator: BeaverTripleGenerator instance
            initial_size: Initial number of triples to generate
        """
        self.generator = generator
        self.triples: List[BeaverTriple] = []
        self.n_nodes = 0
        self.t = 0
        self.initial_size = initial_size
    
    def initialize(self, n_nodes: int, t: int):
        """Initialize pool with triples"""
        self.n_nodes = n_nodes
        self.t = t
        self.replenish(self.initial_size)
    
    def get_triple(self) -> Optional[BeaverTriple]:
        """Get a triple from the pool"""
        if not self.triples:
            return None
        return self.triples.pop()
    
    def replenish(self, num_triples: int):
        """Generate and add new triples to the pool"""
        for _ in range(num_triples):
            triple = self.generator.generate_triple(self.n_nodes, self.t)
            self.triples.append(triple)
    
    def should_replenish(self) -> bool:
        """Check if pool needs replenishing"""
        # Replenish when pool is 30% full (more aggressive to avoid blocking)
        return len(self.triples) < self.initial_size * 3 // 10


class SecureMultiplier:
    """
    High-level interface for secure multiplication using Beaver triples
    """
    
    def __init__(self, triple_pool: BeaverTriplePool, n_nodes: int, t: int,
                 field_size: int = 2**31 - 1,
                 reconstruction_manager: Optional[MPCReconstructionManager] = None,
                 prss_seed: Optional[int] = None):
        """
        Initialize secure multiplier
        Args:
            triple_pool: Pool of Beaver triples
            n_nodes: Number of nodes
            t: Privacy threshold
            field_size: Prime field size
            reconstruction_manager: Optional reconstruction manager for multi-node operations
        """
        self.triple_pool = triple_pool
        self.n_nodes = n_nodes
        self.t = t
        self.field_size = field_size
        self.reconstruction_manager = reconstruction_manager
        # If set, generate Beaver triples deterministically from context/index.
        # This avoids expensive triple pool generation and keeps nodes consistent
        # in multi-process local testing (SIMD-style "packed" triples per chunk).
        self.prss_seed = prss_seed
        
        # Initialize pool if not already initialized
        if triple_pool.n_nodes == 0:
            triple_pool.initialize(n_nodes, t)

    def _prss_u64(self, label: str, context: str, idx: int, extra: int = 0) -> int:
        # Deterministic pseudo-random 64-bit integer derived from (seed,label,context,idx,extra).
        # Uses BLAKE2b for speed and stability.
        h = hashlib.blake2b(digest_size=8)
        h.update(str(self.prss_seed).encode("utf-8"))
        h.update(b"|")
        h.update(label.encode("utf-8"))
        h.update(b"|")
        h.update(context.encode("utf-8"))
        h.update(b"|")
        h.update(str(idx).encode("utf-8"))
        h.update(b"|")
        h.update(str(extra).encode("utf-8"))
        return int.from_bytes(h.digest(), "big")

    def _prss_secret(self, label: str, context: str, idx: int) -> int:
        return self._prss_u64(label, context, idx) % self.field_size

    def _prss_share(self, secret: int, label: str, context: str, idx: int, node_x: int) -> int:
        # Shamir polynomial coefficients: c0=secret, c1..ct derived from PRSS
        # Evaluate at x=node_x.
        y = secret % self.field_size
        x_pow = node_x % self.field_size
        for k in range(1, self.t + 1):
            coeff = self._prss_u64(f"{label}_coeff_{k}", context, idx, extra=k) % self.field_size
            y = (y + (coeff * x_pow) % self.field_size) % self.field_size
            x_pow = (x_pow * node_x) % self.field_size
        return y

    def _prss_triple_for(self, context: str, idx: int, node_id: int, node_x: int) -> Tuple[Share, Share, Share]:
        # Generate (a,b,c=a*b) secrets deterministically, then compute this node's shares.
        a = self._prss_secret("a", context, idx)
        b = self._prss_secret("b", context, idx)
        c = (a * b) % self.field_size
        a_y = self._prss_share(a, "a", context, idx, node_x)
        b_y = self._prss_share(b, "b", context, idx, node_x)
        c_y = self._prss_share(c, "c", context, idx, node_x)
        return (
            Share(x=node_x, y=a_y, node_id=node_id),
            Share(x=node_x, y=b_y, node_id=node_id),
            Share(x=node_x, y=c_y, node_id=node_id),
        )
    
    def multiply(self, share1: Share, share2: Share, node_id: int, 
                context: Optional[str] = None) -> Share:
        """
        Multiply two shares securely using Beaver triple
        Args:
            share1: First share (x)
            share2: Second share (y)
            node_id: ID of this node
            context: Optional context for reconstruction (enables multi-node)
        Returns:
            Share of the product (x * y)
        """
        # Fast path: single-node mode has no privacy (t=0), so Beaver triples
        # provide no additional security but add massive overhead.
        #
        # In the current codebase, the "single-node or simplified multiplication"
        # path already uses local values as a placeholder for reconstruction,
        # so this optimization preserves the exact arithmetic result while
        # removing triple generation / book-keeping costs.
        if self.n_nodes == 1 and self.t == 0 and self.reconstruction_manager is None:
            return Share(
                x=share1.x,
                y=(share1.y * share2.y) % self.field_size,
                node_id=node_id
            )

        # Get triple from pool
        triple = self.triple_pool.get_triple()
        if triple is None:
            # Emergency replenish (more aggressively)
            replenish_size = max(1000, self.triple_pool.initial_size // 10)
            self.triple_pool.replenish(replenish_size)
            triple = self.triple_pool.get_triple()
            if triple is None:
                raise RuntimeError("Failed to get Beaver triple")
        
        # Perform secure multiplication
        if self.reconstruction_manager and context:
            # Multi-node secure multiplication with reconstruction
            result_share = self._multiply_with_reconstruction(
                share1, share2, triple, node_id, context
            )
        else:
            # Single-node or simplified multiplication
            result_share = self._multiply_with_triple(share1, share2, triple, node_id)
        
        # Replenish pool if needed (more aggressively to avoid blocking)
        if self.triple_pool.should_replenish():
            # Replenish with 20% of initial size, but at least 1000
            replenish_size = max(1000, self.triple_pool.initial_size // 5)
            self.triple_pool.replenish(replenish_size)
        
        return result_share

    def multiply_batch(
        self,
        share1_list: List[Share],
        share2_list: List[Share],
        node_id: int,
        context_prefix: Optional[str],
        chunk_timeout: float = 120.0,
    ) -> List[Share]:
        """
        Multiply many share pairs efficiently.

        If multi-node reconstruction is enabled (reconstruction_manager + context_prefix),
        this batches Beaver openings by reconstructing ALL d's and ALL e's in bulk.
        """
        if len(share1_list) != len(share2_list):
            raise ValueError("share1_list and share2_list length mismatch")
        if not share1_list:
            return []

        # Single-node fast path
        if self.n_nodes == 1 and self.t == 0 and self.reconstruction_manager is None:
            return [
                Share(x=a.x, y=(a.y * b.y) % self.field_size, node_id=node_id)
                for a, b in zip(share1_list, share2_list)
            ]

        # If we don't have reconstruction enabled, fall back to scalar multiply
        if not (self.reconstruction_manager and context_prefix):
            out = []
            for i, (a, b) in enumerate(zip(share1_list, share2_list)):
                out.append(self.multiply(a, b, node_id, context=f"{context_prefix}_{i}" if context_prefix else None))
            return out

        a_shares: List[Share] = []
        b_shares: List[Share] = []
        c_shares: List[Share] = []
        d_local: List[Share] = []
        e_local: List[Share] = []

        # Generate triples and form local d/e shares
        node_x = share1_list[0].x
        if self.prss_seed is not None and context_prefix is not None:
            # PRSS-style "packed" triples: deterministic per (context_prefix, idx)
            for i in range(len(share1_list)):
                a_s, b_s, c_s = self._prss_triple_for(context_prefix, i, node_id=node_id, node_x=node_x)
                a_shares.append(a_s)
                b_shares.append(b_s)
                c_shares.append(c_s)
                d_local.append(Share(x=node_x, y=(share1_list[i].y - a_s.y) % self.field_size, node_id=node_id))
                e_local.append(Share(x=node_x, y=(share2_list[i].y - b_s.y) % self.field_size, node_id=node_id))
        else:
            for i in range(len(share1_list)):
                triple = self.triple_pool.get_triple()
                if triple is None:
                    replenish_size = max(1000, self.triple_pool.initial_size // 10)
                    self.triple_pool.replenish(replenish_size)
                    triple = self.triple_pool.get_triple()
                    if triple is None:
                        raise RuntimeError("Failed to get Beaver triple")

                a_s, b_s, c_s = triple.get_for_node(node_id)
                a_shares.append(a_s)
                b_shares.append(b_s)
                c_shares.append(c_s)

                d_local.append(Share(x=share1_list[i].x, y=(share1_list[i].y - a_s.y) % self.field_size, node_id=node_id))
                e_local.append(Share(x=share2_list[i].x, y=(share2_list[i].y - b_s.y) % self.field_size, node_id=node_id))

        # Batch reconstruct d and e
        d_vals, e_vals = self.reconstruction_manager.reconstruct_for_multiplication_batch(
            d_local, e_local, context_prefix=context_prefix, timeout=chunk_timeout
        )

        # Compute outputs
        out: List[Share] = []
        for i in range(len(share1_list)):
            d_recon = d_vals[i]
            e_recon = e_vals[i]
            result_y = (
                c_shares[i].y +
                (d_recon * b_shares[i].y) % self.field_size +
                (e_recon * a_shares[i].y) % self.field_size +
                (d_recon * e_recon) % self.field_size
            ) % self.field_size
            out.append(Share(x=share1_list[i].x, y=result_y, node_id=node_id))

        # Replenish pool if needed (only if we're using the pool)
        if self.prss_seed is None and self.triple_pool.should_replenish():
            replenish_size = max(1000, self.triple_pool.initial_size // 5)
            self.triple_pool.replenish(replenish_size)

        return out

    def multiply_batch_values(
        self,
        y1: np.ndarray,
        y2: np.ndarray,
        *,
        x: int,
        node_id: int,
        context_prefix: str,
        chunk_timeout: float = 120.0,
    ) -> np.ndarray:
        """
        Array-based batch multiplication.
        Avoids constructing Share objects for each scalar multiply.

        Args:
            y1, y2: 1D arrays (same length) of share values modulo field_size
            x: Shamir x-coordinate for this node (usually equals node_id)
            node_id: node id
            context_prefix: unique context prefix for PRSS + openings
        Returns:
            1D numpy array of product share values modulo field_size
        """
        y1 = np.asarray(y1, dtype=np.uint64)
        y2 = np.asarray(y2, dtype=np.uint64)
        if y1.shape != y2.shape:
            raise ValueError("y1 and y2 shape mismatch")
        if y1.ndim != 1:
            y1 = y1.reshape(-1)
            y2 = y2.reshape(-1)
        n = int(y1.size)
        if n == 0:
            return np.zeros((0,), dtype=np.uint64)

        p = int(self.field_size)

        # Single-node fast path
        if self.n_nodes == 1 and self.t == 0 and self.reconstruction_manager is None:
            return (y1 * y2) % p

        # Require reconstruction manager for secure multi-node batching
        if not self.reconstruction_manager:
            # Fallback: do scalar multiplies (slow)
            out = np.empty((n,), dtype=np.uint64)
            for i in range(n):
                out[i] = self.multiply(Share(x=x, y=int(y1[i]), node_id=node_id),
                                       Share(x=x, y=int(y2[i]), node_id=node_id),
                                       node_id=node_id,
                                       context=f"{context_prefix}_{i}").y
            return out

        # Generate a,b,c shares for this node (PRSS preferred)
        a_y = np.empty((n,), dtype=np.uint64)
        b_y = np.empty((n,), dtype=np.uint64)
        c_y = np.empty((n,), dtype=np.uint64)

        if self.prss_seed is not None:
            for i in range(n):
                a_s, b_s, c_s = self._prss_triple_for(context_prefix, i, node_id=node_id, node_x=x)
                a_y[i] = a_s.y
                b_y[i] = b_s.y
                c_y[i] = c_s.y
        else:
            # Pool-based (still works, but slower)
            for i in range(n):
                triple = self.triple_pool.get_triple()
                if triple is None:
                    replenish_size = max(1000, self.triple_pool.initial_size // 10)
                    self.triple_pool.replenish(replenish_size)
                    triple = self.triple_pool.get_triple()
                    if triple is None:
                        raise RuntimeError("Failed to get Beaver triple")
                a_s, b_s, c_s = triple.get_for_node(node_id)
                a_y[i] = a_s.y
                b_y[i] = b_s.y
                c_y[i] = c_s.y

        d_local = (y1 + (p - (a_y % p))) % p
        e_local = (y2 + (p - (b_y % p))) % p

        # Avoid materializing Python int lists: pass numpy arrays directly.
        d_vals, e_vals = self.reconstruction_manager.reconstruct_for_multiplication_batch_values(
            d_vals_local=(d_local % p).astype(np.uint32, copy=False),
            e_vals_local=(e_local % p).astype(np.uint32, copy=False),
            x=x,
            context_prefix=context_prefix,
            timeout=chunk_timeout,
        )
        d = np.asarray(d_vals, dtype=np.uint64) % p
        e = np.asarray(e_vals, dtype=np.uint64) % p

        # result = c + d*b + e*a + d*e  (all mod p)
        res = (c_y % p)
        res = (res + (d * (b_y % p)) % p) % p
        res = (res + (e * (a_y % p)) % p) % p
        res = (res + (d * e) % p) % p
        return res
    
    def _multiply_with_triple(self, share1: Share, share2: Share,
                             triple: BeaverTriple, node_id: int) -> Share:
        """
        Perform secure multiplication using Beaver triple protocol
        
        Protocol:
        1. Compute d = x - a, e = y - b (on shares)
        2. Reconstruct d and e (requires communication - simplified here)
        3. Compute result = c + d*b + e*a + d*e
        
        Args:
            share1: First share (x)
            share2: Second share (y)
            triple: Beaver triple
            node_id: Node ID
        Returns:
            Product share
        """
        # Get triple shares for this node
        a_share, b_share, c_share = triple.get_for_node(node_id)
        
        # Compute d = x - a, e = y - b (on shares)
        d_share = Share(
            x=share1.x,
            y=(share1.y - a_share.y) % self.field_size,
            node_id=node_id
        )
        e_share = Share(
            x=share2.x,
            y=(share2.y - b_share.y) % self.field_size,
            node_id=node_id
        )
        
        # In a full implementation, d and e would be reconstructed across nodes
        # For now, we'll use a simplified approach where we assume we can reconstruct
        # In production, this would require the communication protocol
        
        # Simplified reconstruction (would need actual multi-node communication)
        # For minimal pipeline, we'll use the share values directly
        # This is not fully secure but demonstrates the protocol
        
        # Reconstruct d and e (simplified - would need shares from all nodes)
        d_recon = d_share.y  # Placeholder - would reconstruct from all nodes
        e_recon = e_share.y  # Placeholder - would reconstruct from all nodes
        
        # Compute result share: c + d*b + e*a + d*e
        result_y = (
            c_share.y +
            (d_recon * b_share.y) % self.field_size +
            (e_recon * a_share.y) % self.field_size +
            (d_recon * e_recon) % self.field_size
        ) % self.field_size
        
        result_share = Share(
            x=share1.x,
            y=result_y,
            node_id=node_id
        )
        
        return result_share
    
    def _multiply_with_reconstruction(self, share1: Share, share2: Share,
                                     triple: BeaverTriple, node_id: int,
                                     context: str) -> Share:
        """
        Perform secure multiplication using Beaver triple protocol with multi-node reconstruction
        
        Protocol:
        1. Compute d = x - a, e = y - b (on shares)
        2. Reconstruct d and e across nodes using network
        3. Compute result = c + d*b + e*a + d*e
        
        Args:
            share1: First share (x)
            share2: Second share (y)
            triple: Beaver triple
            node_id: Node ID
            context: Context identifier for share exchange
        Returns:
            Product share
        """
        if not self.reconstruction_manager:
            # Fallback to simplified version
            return self._multiply_with_triple(share1, share2, triple, node_id)
        
        # Get triple shares for this node
        a_share, b_share, c_share = triple.get_for_node(node_id)
        
        # Compute d = x - a, e = y - b (on shares)
        d_share = Share(
            x=share1.x,
            y=(share1.y - a_share.y) % self.field_size,
            node_id=node_id
        )
        e_share = Share(
            x=share2.x,
            y=(share2.y - b_share.y) % self.field_size,
            node_id=node_id
        )
        
        # Reconstruct d and e across nodes using network
        try:
            d_recon, e_recon = self.reconstruction_manager.reconstruct_for_multiplication(
                [d_share], [e_share], context
            )
            # Debug: Show that multi-node reconstruction is working
            if hasattr(self, '_debug_log') and self._debug_log:
                print(f"  [Multi-Node] Reconstructed d and e using context '{context}'")
        except Exception as e:
            # If reconstruction fails, fall back to simplified version
            # This can happen if not enough nodes are connected
            if hasattr(self, '_debug_log') and self._debug_log:
                print(f"  [Fallback] Reconstruction failed for '{context}': {e}")
            d_recon = d_share.y
            e_recon = e_share.y
        
        # Compute result share: c + d*b + e*a + d*e
        result_y = (
            c_share.y +
            (d_recon * b_share.y) % self.field_size +
            (e_recon * a_share.y) % self.field_size +
            (d_recon * e_recon) % self.field_size
        ) % self.field_size
        
        result_share = Share(
            x=share1.x,
            y=result_y,
            node_id=node_id
        )
        
        return result_share

