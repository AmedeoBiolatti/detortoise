import jax
from jax.experimental import pallas as pl


def _dequantize_kernel(
        quantized_ref,
        scales_ref,
        nf4_levels_ref,
        o_ref
):
    i = pl.program_id(0)

    for j in range(32):
        q = quantized_ref[i * 32 + j]
        v1 = nf4_levels_ref[q & 0xF]
        v2 = nf4_levels_ref[(q >> 4) & 0xF]

        o_ref[i * 64 + 2 * j] = v1 * scales_ref[i]
        o_ref[i * 64 + 2 * j + 1] = v2 * scales_ref[i]


def dequantize(
        quantized,
        scales,
        original_shape,
        # block_size,
        num_blocks,
        pad_length,
        nf4_levels,
        dtype="float32"
) -> jax.Array:
    out = pl.pallas_call(
        _dequantize_kernel,
        out_shape=jax.ShapeDtypeStruct((quantized.shape[0] * 2,), dtype),
        grid=(num_blocks,),
    )(quantized, scales, nf4_levels)
    if pad_length > 0:
        out = out[:-pad_length]
    out = jax.numpy.reshape(out, original_shape)
    return out
