"""
Secure Softmax and Cross-Entropy Loss Operations
Implements secure softmax and cross-entropy loss for classification
Uses numerically stable approximations suitable for MPC
"""

from typing import List, Tuple
from ml_training.secret_sharing import Share
from ml_training.beaver_triples import SecureMultiplier
from ml_training.secure_division import SecureDivider
import numpy as np


class SecureSoftmax:
    """
    Performs secure softmax operations on secret-shared logits
    Uses polynomial approximation for exp(x) in secure computation
    """
    
    def __init__(self, multiplier: SecureMultiplier, divider: SecureDivider, 
                 field_size: int = 2**32 - 5, scale_factor: int = 1000,
                 temperature: float = 2.0):
        """
        Initialize secure softmax
        Args:
            multiplier: SecureMultiplier instance
            divider: SecureDivider instance
            field_size: Prime field size
            scale_factor: Scaling factor for integer representation
            temperature: Temperature scaling factor (default 2.0 for softer probabilities)
                         Higher temperature = softer probabilities (less confident)
                         Lower temperature = sharper probabilities (more confident)
        """
        self.multiplier = multiplier
        self.divider = divider
        self.field_size = field_size
        self.SCALE_FACTOR = scale_factor
        self.temperature = temperature

        # Precompute inverses for fixed-point polynomial coefficients (public scalars)
        p = int(self.field_size)
        self._inv_scale = pow(int(self.SCALE_FACTOR) % p, p - 2, p)
        self._inv2 = pow(2, p - 2, p)
        self._inv3 = pow(3, p - 2, p)
        self._inv5 = pow(5, p - 2, p)
        self._inv7 = pow(7, p - 2, p)

    def _fp_mul(self, a: Share, b: Share, node_id: int, context: str) -> Share:
        """
        Fixed-point multiply of two SCALE_FACTOR-scaled shares:
            (a*b)/SCALE_FACTOR  (result is SCALE_FACTOR-scaled)
        """
        # Use multiplier's fixed-point multiply so "A" mode truncation applies consistently.
        return self.multiplier.multiply_fixed_point(
            a,
            b,
            node_id=node_id,
            scale_factor=int(self.SCALE_FACTOR),
            context=context,
        )

    def _fp_add(self, a: Share, b: Share, node_id: int) -> Share:
        return Share(x=a.x, y=(int(a.y) + int(b.y)) % int(self.field_size), node_id=node_id)

    def _fp_sub(self, a: Share, b: Share, node_id: int) -> Share:
        return Share(x=a.x, y=(int(a.y) - int(b.y)) % int(self.field_size), node_id=node_id)

    def _exp_poly(self, x: Share, node_id: int, context: str) -> Share:
        """
        MPC-safe exp approximation (fixed-point) using a 5th order Taylor polynomial:
          exp(x) ≈ 1 + x + x^2/2 + x^3/6 + x^4/24 + x^5/120

        This avoids reading Share values as plaintext. Accuracy depends on logits being
        reasonably small in magnitude (which must be enforced by the numeric pipeline).
        """
        p = int(self.field_size)
        one = Share(x=x.x, y=int(self.SCALE_FACTOR) % p, node_id=node_id)

        x2 = self._fp_mul(x, x, node_id, context=f"{context}_x2")
        x3 = self._fp_mul(x2, x, node_id, context=f"{context}_x3")
        x4 = self._fp_mul(x3, x, node_id, context=f"{context}_x4")
        x5 = self._fp_mul(x4, x, node_id, context=f"{context}_x5")

        # IMPORTANT: do NOT use modular inverses for division (creates field fractions).
        # Use truncating scalar division to preserve integer fixed-point semantics.
        t2 = self.divider.secure_scalar_divide(x2, 2, node_id, context=f"{context}_div2")
        t3 = self.divider.secure_scalar_divide(x3, 6, node_id, context=f"{context}_div6")
        t4 = self.divider.secure_scalar_divide(x4, 24, node_id, context=f"{context}_div24")
        t5 = self.divider.secure_scalar_divide(x5, 120, node_id, context=f"{context}_div120")

        out = self._fp_add(one, x, node_id)
        out = self._fp_add(out, t2, node_id)
        out = self._fp_add(out, t3, node_id)
        out = self._fp_add(out, t4, node_id)
        out = self._fp_add(out, t5, node_id)
        return out

    def _atanh_series(self, y: Share, node_id: int, context: str, terms: int = 5) -> Share:
        """
        Approximate atanh(y) via series:
          atanh(y) = y + y^3/3 + y^5/5 + ...  (|y|<1)
        returns SCALE_FACTOR-scaled share.
        """
        p = int(self.field_size)
        out = Share(x=y.x, y=int(y.y) % p, node_id=node_id)
        y2 = self._fp_mul(y, y, node_id, context=f"{context}_y2")
        ypow = Share(x=y.x, y=int(y.y) % p, node_id=node_id)  # y^(2k+1)
        denom = 1
        for k in range(1, terms):
            # y^(2k+1) = y^(2k-1) * y^2
            ypow = self._fp_mul(ypow, y2, node_id, context=f"{context}_ypow_{k}")
            denom = 2 * k + 1
            term = self.divider.secure_scalar_divide(ypow, denom, node_id, context=f"{context}_div_{denom}")
            out = self._fp_add(out, term, node_id)
        return out

    def _log_fp(self, x: Share, node_id: int, context: str) -> Share:
        """
        Approximate ln(x) for x in (0,1] (fixed-point scaled) using:
          ln(x) = 2 * atanh((x-1)/(x+1))

        Note: This is approximate and requires x not too close to 0 for good accuracy.
        """
        p = int(self.field_size)
        one = Share(x=x.x, y=int(self.SCALE_FACTOR) % p, node_id=node_id)
        num = self._fp_sub(x, one, node_id)
        den = self._fp_add(x, one, node_id)
        y = self.divider.secure_divide_shares(num, den, node_id=node_id, context=f"{context}_div")
        at = self._atanh_series(y, node_id, context=f"{context}_atanh", terms=6)
        two = 2 % p
        return Share(x=x.x, y=(int(at.y) * two) % p, node_id=node_id)
    
    def softmax(self, logits: List[Share], node_id: int, context: str = "softmax") -> List[Share]:
        """
        Compute softmax of logits: softmax(x_i) = exp(x_i/T - max(x)/T) / sum(exp(x_j/T - max(x)/T))
        Uses max subtraction for numerical stability and temperature scaling to reduce overconfidence
        
        Args:
            logits: List of shares representing logits [num_classes]
            node_id: Node ID
            context: Context for secure operations
        Returns:
            List of shares representing softmax probabilities [num_classes]
        """
        p = int(self.field_size)
        if len(logits) == 0:
            return []

        # Temperature scaling: logits := logits / T
        # IMPORTANT: do not use modular inverse scaling (field division) in fixed-point.
        # In privacy_mode, SecureDivider.secure_scalar_divide uses opener truncation.
        if self.temperature != 1.0:
            temp_int = int(round(float(self.temperature)))
            if abs(float(self.temperature) - float(temp_int)) > 1e-9:
                raise ValueError("Non-integer temperature is not supported in fixed-point privacy_mode")
            logits = [
                self.divider.secure_scalar_divide(l, temp_int, node_id, context=f"{context}_temp_{i}")
                for i, l in enumerate(logits)
            ]

        # Stabilize without secure max: subtract the (secret) mean.
        # softmax(x) == softmax(x - c) for any constant c.
        # mean is linear, so we can compute it securely without comparisons.
        mean = Share(x=logits[0].x, y=0, node_id=node_id)
        for l in logits:
            mean = Share(x=mean.x, y=(int(mean.y) + int(l.y)) % p, node_id=node_id)
        mean = self.divider.secure_scalar_divide(mean, len(logits), node_id, context=f"{context}_mean")
        logits = [Share(x=l.x, y=(int(l.y) - int(mean.y)) % p, node_id=node_id) for l in logits]

        # exp and sumexp (all fixed-point scaled)
        exp_logits = [self._exp_poly(l, node_id=node_id, context=f"{context}_exp_{i}") for i, l in enumerate(logits)]
        sumexp = Share(x=exp_logits[0].x, y=0, node_id=node_id)
        for e in exp_logits:
            sumexp = Share(x=sumexp.x, y=(int(sumexp.y) + int(e.y)) % p, node_id=node_id)

        # probs = exp / sumexp  (fixed-point scaled)
        probs = [
            self.divider.secure_divide_shares(e, sumexp, node_id=node_id, context=f"{context}_div_{i}")
            for i, e in enumerate(exp_logits)
        ]
        return probs
    
    def cross_entropy_loss(self, logits: List[Share], target: List[Share], 
                          node_id: int, context: str = "ce_loss") -> Share:
        """
        Compute cross-entropy loss: -log(softmax(logits)[true_class])
        Uses log-softmax trick for numerical stability
        
        Args:
            logits: List of shares representing logits [num_classes]
            target: List of shares representing one-hot target [num_classes]
            node_id: Node ID
            context: Context for secure operations
        Returns:
            Share of cross-entropy loss (scaled)
        """
        # MPC-safe CE:  L = - sum_i (y_i * log(p_i))
        # - y is one-hot encoded but secret-shared; we must NOT "find true class" from shares.
        # - probabilities p_i are secret-shared outputs of softmax.
        p_mod = int(self.field_size)
        probs = self.softmax(logits, node_id, f"{context}_softmax")
        if not probs:
            return Share(x=logits[0].x if logits else 0, y=0, node_id=node_id)

        acc = Share(x=probs[0].x, y=0, node_id=node_id)
        for i, (p_i, y_i) in enumerate(zip(probs, target)):
            logp = self._log_fp(p_i, node_id=node_id, context=f"{context}_log_{i}")
            # y_i is SCALE-scaled (0 or SCALE). fp-mul yields y_i/scale * logp = indicator * logp.
            term = self._fp_mul(y_i, logp, node_id=node_id, context=f"{context}_mask_{i}")
            acc = Share(x=acc.x, y=(int(acc.y) + int(term.y)) % p_mod, node_id=node_id)

        # loss = -acc
        return Share(x=acc.x, y=(-int(acc.y)) % p_mod, node_id=node_id)
    
    def cross_entropy_loss_gradient(self, logits: List[Share], target: List[Share],
                                   node_id: int, context: str = "ce_grad") -> List[Share]:
        """
        Compute gradient of cross-entropy loss w.r.t. logits
        Gradient = softmax(logits) - target (one-hot)
        
        This is the key advantage: gradient is simple!
        
        Args:
            logits: List of shares representing logits [num_classes]
            target: List of shares representing one-hot target [num_classes]
            node_id: Node ID
            context: Context for secure operations
        Returns:
            List of shares representing gradients [num_classes]
        """
        # MPC-safe gradient:
        # - Let p = softmax(logits / T).
        # - Then dL/dlogits = (p - y) / T.
        # - p and y are SCALE-scaled, so (p-y) is SCALE-scaled and we apply a public 1/T factor.
        p_mod = int(self.field_size)
        probs = self.softmax(logits, node_id, f"{context}_softmax")
        grads: List[Share] = []
        for i, (p_i, y_i) in enumerate(zip(probs, target)):
            grads.append(Share(x=p_i.x, y=(int(p_i.y) - int(y_i.y)) % p_mod, node_id=node_id))
        if self.temperature != 1.0 and grads:
            temp_int = int(round(float(self.temperature)))
            if abs(float(self.temperature) - float(temp_int)) > 1e-9:
                raise ValueError("Non-integer temperature is not supported in fixed-point privacy_mode")
            grads = [
                self.divider.secure_scalar_divide(g, temp_int, node_id, context=f"{context}_tempgrad_{i}")
                for i, g in enumerate(grads)
            ]
        return grads
