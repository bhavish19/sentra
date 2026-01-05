"""
Secret Sharing Implementation
Shamir Secret Sharing and Packed Shamir Secret Sharing (PSS)
"""

from typing import List, Tuple
from dataclasses import dataclass
import random


@dataclass
class Share:
    """Represents a secret share"""
    x: int  # Evaluation point
    y: int  # Share value
    node_id: int  # Node that holds this share


class ShamirSecretSharing:
    """
    Shamir Secret Sharing implementation
    Splits a secret into n shares, requires t+1 shares to reconstruct
    """
    
    def __init__(self, field_size: int = 2**31 - 1):
        """
        Initialize Shamir Secret Sharing
        Args:
            field_size: Prime field size (default: 2^31 - 1)
        """
        self.field_size = field_size
    
    def share(self, secret: int, n: int, t: int) -> List[Share]:
        """
        Split secret into n shares with threshold t
        Args:
            secret: Secret value to share
            n: Number of shares
            t: Privacy threshold (need t+1 shares to reconstruct)
        Returns:
            List of n shares
        """
        if t >= n:
            raise ValueError("Threshold t must be less than n")
        
        # Ensure secret is in field
        secret = secret % self.field_size
        
        # Generate random polynomial coefficients
        # f(x) = secret + a1*x + a2*x^2 + ... + at*x^t
        coefficients = [secret]
        for _ in range(t):
            coefficients.append(random.randint(0, self.field_size - 1))
        
        # Generate shares at points 1, 2, ..., n
        shares = []
        for i in range(1, n + 1):
            y = self._evaluate_polynomial(coefficients, i)
            shares.append(Share(x=i, y=y, node_id=i))
        
        return shares
    
    def reconstruct(self, shares: List[Share]) -> int:
        """
        Reconstruct secret from shares using Lagrange interpolation
        Args:
            shares: List of at least t+1 shares
        Returns:
            Reconstructed secret
        """
        if len(shares) < 2:
            raise ValueError("Need at least 2 shares to reconstruct")
        
        # Use Lagrange interpolation
        secret = 0
        for i, share_i in enumerate(shares):
            numerator = 1
            denominator = 1
            
            for j, share_j in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-share_j.x)) % self.field_size
                    denominator = (denominator * (share_i.x - share_j.x)) % self.field_size
            
            # Compute Lagrange basis polynomial value
            lagrange_basis = (numerator * self._mod_inverse(denominator)) % self.field_size
            secret = (secret + share_i.y * lagrange_basis) % self.field_size
        
        return secret
    
    def _evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at point x"""
        result = 0
        power = 1
        for coeff in coefficients:
            result = (result + coeff * power) % self.field_size
            power = (power * x) % self.field_size
        return result
    
    def _mod_inverse(self, a: int) -> int:
        """Compute modular inverse using extended Euclidean algorithm"""
        if a < 0:
            a = a % self.field_size
        
        # Extended Euclidean algorithm
        old_r, r = a, self.field_size
        old_s, s = 1, 0
        
        while r != 0:
            quotient = old_r // r
            old_r, r = r, old_r - quotient * r
            old_s, s = s, old_s - quotient * s
        
        if old_r != 1:
            raise ValueError(f"{a} has no modular inverse modulo {self.field_size}")
        
        return old_s % self.field_size


class PackedShamirSecretSharing:
    """
    Packed Shamir Secret Sharing (PSS)
    Packs multiple secrets into a single polynomial for SIMD-style parallelism
    """
    
    def __init__(self, field_size: int = 2**31 - 1):
        """
        Initialize Packed Shamir Secret Sharing
        Args:
            field_size: Prime field size
        """
        self.field_size = field_size
        self.shamir = ShamirSecretSharing(field_size)
    
    def pack_share(self, shares: List[Share], packing_factor: int) -> List[Share]:
        """
        Pack multiple secrets into packed shares
        Args:
            shares: List of shares (one per secret)
            packing_factor: Number of secrets to pack per share
        Returns:
            List of packed shares
        """
        if len(shares) == 0:
            return []
        
        # Group shares by node_id
        shares_by_node: dict = {}
        for share in shares:
            if share.node_id not in shares_by_node:
                shares_by_node[share.node_id] = []
            shares_by_node[share.node_id].append(share)
        
        # Create packed shares
        packed_shares = []
        for node_id, node_shares in shares_by_node.items():
            # Pack values at this node
            packed_y = 0
            for i, share in enumerate(node_shares[:packing_factor]):
                packed_y = (packed_y + share.y * (self.field_size ** i)) % self.field_size
            
            packed_shares.append(Share(
                x=node_shares[0].x,
                y=packed_y,
                node_id=node_id
            ))
        
        return packed_shares
    
    def unpack_share(self, packed_shares: List[Share], packing_factor: int) -> List[Share]:
        """
        Unpack shares back to individual secrets
        Args:
            packed_shares: Packed shares
            packing_factor: Number of secrets packed per share
        Returns:
            List of unpacked shares
        """
        unpacked = []
        for packed_share in packed_shares:
            # Extract individual values
            y = packed_share.y
            for i in range(packing_factor):
                value = y % self.field_size
                unpacked.append(Share(
                    x=packed_share.x,
                    y=value,
                    node_id=packed_share.node_id
                ))
                y = y // self.field_size
        
        return unpacked


