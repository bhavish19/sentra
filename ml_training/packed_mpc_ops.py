"""
Packed-Shamir MPC primitives.

This module provides a practical way to do secret×secret multiplication on packed shares
WITHOUT a trusted opener, by:
  packed -> (distributed) unpack to k lane Shamir shares
  lane-wise Beaver multiplication (existing SecureMultiplier)
  (distributed) pack back to packed shares

This preserves the packed representation between ops (useful for "SIMD-at-rest"),
while reusing the existing Beaver triple machinery for correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict

import time
import random
import hashlib

import numpy as np

from ml_training.secret_sharing import Share, ShamirSecretSharing, PackedShamirSecretSharing


def _lagrange_coeffs_at(x0: int, xs: List[int], p: int) -> List[int]:
    """Return lambdas such that f(x0)=sum_i lambda_i * f(xs[i]) for deg < len(xs)."""
    x0 = int(x0) % p
    xs = [int(x) % p for x in xs]
    out: List[int] = []
    for i, xi in enumerate(xs):
        num = 1
        den = 1
        for j, xj in enumerate(xs):
            if i == j:
                continue
            num = (num * (x0 - xj)) % p
            den = (den * (xi - xj)) % p
        inv_den = pow(den % p, p - 2, p)
        out.append((num * inv_den) % p)
    return out


def _secret_point_basis_coeffs_at(i_point: int, secret_xs: List[int], p: int) -> List[int]:
    """L_j(i): basis coeffs for interpolation from secret_xs to point i."""
    i_point = int(i_point) % p
    sx = [int(s) % p for s in secret_xs]
    coeffs: List[int] = []
    for j, sj in enumerate(sx):
        num = 1
        den = 1
        for m, sm in enumerate(sx):
            if m == j:
                continue
            num = (num * (i_point - sm)) % p
            den = (den * (sj - sm)) % p
        coeffs.append((num * pow(den % p, p - 2, p)) % p)
    return coeffs


def _P_at(i_point: int, secret_xs: List[int], p: int) -> int:
    """P(x)=∏(x - secret_xs[j]) evaluated at i_point."""
    i_point = int(i_point) % p
    out = 1
    for sx in secret_xs:
        out = (out * (i_point - (int(sx) % p))) % p
    return out


def _det_u32_seed(*parts: str) -> int:
    """
    Deterministic 32-bit seed from string parts (stable across processes).
    """
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest(), "big") & 0xFFFFFFFF


@dataclass
class PackedMPCOps:
    network: any
    n_nodes: int
    t: int
    field_size: int
    secret_xs: Optional[List[int]] = None

    def __post_init__(self):
        self.n_nodes = int(self.n_nodes)
        self.t = int(self.t)
        self.field_size = int(self.field_size)
        self.shamir = ShamirSecretSharing(self.field_size)
        self.pss = PackedShamirSecretSharing(self.field_size)

    def _secret_xs_for_k(self, k: int) -> List[int]:
        if self.secret_xs is not None:
            if len(self.secret_xs) != int(k):
                raise ValueError("secret_xs length mismatch")
            return list(self.secret_xs)
        return [-(i + 1) for i in range(int(k))]

    def unpack_packed_to_lane_shares(
        self,
        *,
        packed_share: Share,
        k: int,
        context: str,
        timeout: float = 60.0,
    ) -> List[Share]:
        """
        Convert one packed share (per node) into k Shamir shares (one per lane secret).
        No opening; uses distributed reshare of linear combinations.
        """
        p = int(self.field_size)
        node_id = int(self.network.node_id)
        k = int(k)
        need = int(self.t) + int(k)
        if need > int(self.n_nodes):
            raise ValueError("k exceeds n-t")

        chosen_nodes = list(range(1, need + 1))  # deterministic subset
        xs = chosen_nodes[:]  # x-points are 1..need
        secret_xs = self._secret_xs_for_k(k)

        # Compute coefficients for each lane secret at secret_xs[j]
        # secret_j = sum_i lambda_i(secret_xs[j]) * packed_y_i
        lambdas_by_lane: List[List[int]] = []
        for sx in secret_xs:
            lambdas_by_lane.append(_lagrange_coeffs_at(sx, xs, p))

        # Local contributions m_{i,j} = lambda_i(sx_j) * y_i
        m_vals: List[int] = [0] * k
        if node_id in chosen_nodes:
            idx = chosen_nodes.index(node_id)
            for j in range(k):
                m_vals[j] = (int(lambdas_by_lane[j][idx]) * int(packed_share.y)) % p

        # Each node Shamir-shares each m_vals[j] to all nodes, then sends per-recipient vector.
        per_recipient = {rid: np.zeros((k,), dtype=np.uint32) for rid in range(1, self.n_nodes + 1)}
        for j in range(k):
            shares = self.shamir.share(int(m_vals[j]) % p, self.n_nodes, self.t)
            for s in shares:
                per_recipient[int(s.node_id)][j] = np.uint32(int(s.y) & 0xFFFFFFFF)

        # Send to all recipients (including those not in chosen_nodes; they just sum zeros+noise)
        for rid in range(1, self.n_nodes + 1):
            if rid == node_id:
                continue
            self.network.channel.send_vector(int(rid), context, x=int(rid), values=per_recipient[int(rid)])

        # Receiver sums vectors from all chosen senders (and includes own)
        start = time.time()
        while time.time() - start < timeout:
            recv = self.network.channel.get_received_vector(context)
            # ensure our own contribution is included
            recv[node_id] = {"x": int(node_id), "values": per_recipient[node_id]}
            if all(sid in recv for sid in chosen_nodes):
                break
            time.sleep(0.01)

        # Sum k-lane shares for this receiver
        lane_y = np.zeros((k,), dtype=np.uint64)
        for sid in chosen_nodes:
            vec = np.asarray(recv[int(sid)]["values"], dtype=np.uint64)
            lane_y = (lane_y + vec) % np.uint64(p)

        try:
            self.network.channel.clear_vector(context)
        except Exception:
            pass

        return [Share(x=node_id, y=int(lane_y[j] % p), node_id=node_id) for j in range(k)]

    def pack_lane_shares_to_packed(
        self,
        *,
        lane_shares: List[Share],
        k: int,
        context: str,
        timeout: float = 60.0,
    ) -> Share:
        """
        Convert k lane Shamir shares (held by each node) back into ONE packed share per node.
        No opening; uses a distributed "send contributions to each target node" protocol.
        """
        p = int(self.field_size)
        node_id = int(self.network.node_id)
        k = int(k)
        if len(lane_shares) != k:
            raise ValueError("lane_shares length mismatch")

        secret_xs = self._secret_xs_for_k(k)
        d = int(self.t) + int(k) - 1

        # Base reconstruction uses t+1 nodes (for each secret lane)
        base_nodes = list(range(1, int(self.t) + 2))  # 1..t+1
        lam0 = _lagrange_coeffs_at(0, base_nodes, p)  # for Shamir secrets at x=0

        # Each sender must choose ONE random degree-(t-1) polynomial q(x) (per context),
        # then evaluate it at all targets. If we sample fresh randomness per target,
        # the resulting mask will not be a polynomial in x and the packed share becomes invalid.
        if self.t == 0:
            q_coeffs = [0]
        else:
            rng = random.Random(_det_u32_seed("pack_mask_q", str(context), str(node_id)))
            q_coeffs = [rng.randrange(0, p) for _ in range(self.t)]  # degree t-1

        def _q_eval(x: int) -> int:
            x = int(x) % p
            out = 0
            power = 1
            for c in q_coeffs:
                out = (out + int(c) * power) % p
                power = (power * x) % p
            return out

        # Precompute per-target coefficients L_j(i) and P(i)
        # Each sender computes total_contrib_to_i = base_contrib_to_i + mask_contrib_to_i and sends it.
        for target in range(1, self.n_nodes + 1):
            coeffs = _secret_point_basis_coeffs_at(target, secret_xs, p)  # length k
            # sum_j coeffs[j] * f_j(node_id)
            s_local = 0
            for j in range(k):
                s_local = (s_local + int(coeffs[j]) * int(lane_shares[j].y)) % p

            base_contrib = 0
            if node_id in base_nodes:
                base_contrib = (int(lam0[base_nodes.index(node_id)]) * int(s_local)) % p

            # Mask: r_i(x)=P(x)*q_i(x), q_i degree t-1 random. Sum across nodes yields degree d and zeros at secret_xs.
            Pval = _P_at(target, secret_xs, p)
            q_eval = _q_eval(target) if self.t != 0 else 0
            mask_contrib = (int(Pval) * int(q_eval)) % p

            total = (int(base_contrib) + int(mask_contrib)) % p

            # Send to target under a per-target context
            ctx_t = f"{context}_to_{target}"
            if target != node_id:
                self.network.channel.send_vector(int(target), ctx_t, x=int(node_id), values=np.asarray([int(total) & 0xFFFFFFFF], dtype=np.uint32))

        # Receive contributions addressed to us and sum them
        ctx_me = f"{context}_to_{node_id}"
        start = time.time()
        while time.time() - start < timeout:
            recv = self.network.channel.get_received_vector(ctx_me)
            # add our own contribution by looking at the message we would have sent to self (recompute)
            # (Simpler: send to self is not required; we just recompute and inject sender entry)
            if len(recv) >= (self.n_nodes - 1):
                break
            time.sleep(0.01)

        # Sum from all nodes (including self)
        total_sum = 0
        # Include received from others
        for sid, item in recv.items():
            vec = item.get("values")
            if vec is None:
                continue
            total_sum = (total_sum + int(np.asarray(vec, dtype=np.uint32)[0])) % p

        # Add our own contribution (computed during the send loop above) by re-running for target=node_id
        coeffs = _secret_point_basis_coeffs_at(node_id, secret_xs, p)
        s_local = 0
        for j in range(k):
            s_local = (s_local + int(coeffs[j]) * int(lane_shares[j].y)) % p
        base_contrib = 0
        if node_id in base_nodes:
            base_contrib = (int(lam0[base_nodes.index(node_id)]) * int(s_local)) % p
        Pval = _P_at(node_id, secret_xs, p)
        q_eval = _q_eval(node_id) if self.t != 0 else 0
        mask_contrib = (int(Pval) * int(q_eval)) % p
        total_self = (int(base_contrib) + int(mask_contrib)) % p
        total_sum = (total_sum + int(total_self)) % p

        try:
            self.network.channel.clear_vector(ctx_me)
        except Exception:
            pass

        return Share(x=node_id, y=int(total_sum) % p, node_id=node_id)

    def packed_mul(
        self,
        *,
        a_packed: Share,
        b_packed: Share,
        k: int,
        multiplier,
        context: str,
        timeout: float = 120.0,
    ) -> Share:
        """
        Secret×secret multiplication for packed shares, without opener reconstruction.
        """
        k = int(k)
        # Unpack to lane shares (Shamir, degree t)
        a_lanes = self.unpack_packed_to_lane_shares(packed_share=a_packed, k=k, context=f"{context}_unpack_a", timeout=timeout)
        b_lanes = self.unpack_packed_to_lane_shares(packed_share=b_packed, k=k, context=f"{context}_unpack_b", timeout=timeout)

        # Lane-wise Beaver multiplication (degree t, no opener).
        # IMPORTANT: use multiplier.multiply_batch with a context_prefix so the triple dealer
        # can serve consistent triples across nodes. The scalar multiply() path uses the local
        # triple pool, which is not coordinated across processes.
        c_lanes = multiplier.multiply_batch(
            a_lanes,
            b_lanes,
            node_id=int(self.network.node_id),
            context_prefix=f"{context}_mul",
            chunk_timeout=float(timeout),
        )

        # Pack lane products back into one packed share
        c_packed = self.pack_lane_shares_to_packed(lane_shares=c_lanes, k=k, context=f"{context}_pack_c", timeout=timeout)
        return c_packed

