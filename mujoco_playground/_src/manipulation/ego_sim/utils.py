import jax
import jax.numpy as jp


def euler_to_mat(rpy: jax.Array) -> jax.Array:
    """Convert Euler angles to rotation matrix - ultra-optimized for JAX."""
    # Single vectorized sincos computation (faster than separate sin/cos)
    cos_vals, sin_vals = jp.cos(rpy), jp.sin(rpy)

    # Direct unpacking avoids indexing overhead
    cx, cy, cz = cos_vals
    sx, sy, sz = sin_vals

    # Fused precomputation in single operations
    cz_cy, sz_cy = cz * cy, sz * cy
    cz_sy, sz_sy = cz * sy, sz * sy
    cy_sx, cy_cx = cy * sx, cy * cx
    sz_cx, cz_cx = sz * cx, cz * cx
    sz_sx, cz_sx = sz * sx, cz * sx

    # Direct matrix construction without intermediate arrays
    return jp.stack(
        [
            jp.stack([cz_cy, cz_sy * sx - sz_cx, cz_sy * cx + sz_sx]),
            jp.stack([sz_cy, sz_sy * sx + cz_cx, sz_sy * cx - cz_sx]),
            jp.stack([-sy, cy_sx, cy_cx]),
        ]
    )


def mat_to_quat(R: jax.Array) -> jax.Array:
    """Convert rotation matrix to quaternion - ultra-optimized for JAX."""
    # Extract diagonal and trace in one operation
    diag = jp.diag(R)
    trace = jp.sum(diag)

    # Vectorized square root computations using rsqrt for better performance
    sqrt_vals = (
        jp.sqrt(
            jp.maximum(
                0.0,
                jp.stack(
                    [
                        1.0 + trace,
                        1.0 + 2 * diag[0] - trace,
                        1.0 + 2 * diag[1] - trace,
                        1.0 + 2 * diag[2] - trace,
                    ]
                ),
            )
        )
        * 0.5
    )

    # Vectorized sign extraction
    signs = jp.stack(
        [
            1.0,
            jp.sign(R[2, 1] - R[1, 2]),
            jp.sign(R[0, 2] - R[2, 0]),
            jp.sign(R[1, 0] - R[0, 1]),
        ]
    )

    # Fused quaternion computation
    q = sqrt_vals * signs

    # Fast normalization with precomputed reciprocal
    norm_sq = jp.sum(q * q)
    inv_norm = jp.where(norm_sq > 1e-16, 1 / jp.sqrt(norm_sq), 0.0)
    q_normalized = q * inv_norm

    # Fallback to identity quaternion if norm too small
    return jp.where(norm_sq > 1e-16, q_normalized, jp.array([1.0, 0.0, 0.0, 0.0]))
