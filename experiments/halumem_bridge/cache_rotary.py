"""Inverse rotary phase at known nonnegative absolute cache positions."""
import mlx.core as mx


def inverse_rotary(rope, keys, start):
    if start < 0:
        raise ValueError("Expected a nonnegative absolute key position.")
    # A reflection conjugates a positive planar rotation to its inverse.
    # The channels outside the actual RoPE subspace remain unchanged.
    width=keys.shape[-1]
    dim=rope.dims
    assert dim%2==0 and dim<=width
    channel=mx.arange(width)
    second=(channel%2==1) if rope.traditional else (channel>=dim//2)
    sign=mx.where((channel<dim)&second,-1.,1.).astype(mx.float32)
    return sign*rope(sign*keys.astype(mx.float32),offset=start)
