import keras

# TODO nested scale
NF4_LEVELS = keras.ops.array([
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0
], dtype="float32")


def encode_to_nf4(array, block_size=64):
    import numpy as np
    array = keras.ops.convert_to_numpy(array)

    original_shape = array.shape
    array_flat = array.flatten()

    # Pad array to make it divisible by block_size
    pad_length = (block_size - len(array_flat) % block_size) % block_size
    if pad_length > 0:
        array_flat = np.concatenate([array_flat, np.zeros(pad_length)])

    # Reshape into blocks
    num_blocks = len(array_flat) // block_size
    blocks = array_flat.reshape(num_blocks, block_size)

    # Storage for quantized data
    blocks_max_abs = np.max(np.abs(blocks), axis=1)  # (num_blocks)
    scales = np.where(blocks_max_abs > 0, blocks_max_abs / np.max(np.abs(NF4_LEVELS)), 1.0)
    scales_ = np.reshape(scales, (-1, 1))
    normalized_blocks = blocks / np.where(scales_ > 0, scales_, 1.)
    normalized_blocks = np.reshape(normalized_blocks, (num_blocks, block_size, 1))
    nf4_levels_ = np.reshape(NF4_LEVELS, (1, 1, -1))
    distances = np.abs(normalized_blocks - nf4_levels_)  # (num_blocks, block_size, nf4_levels)
    closest_idx = np.argmin(distances, axis=2)
    quantized_blocks = closest_idx
    #

    j = np.arange(0, block_size // 2)
    packed_data = (quantized_blocks[:, 2 * j] & 0xF) | ((quantized_blocks[:, 2 * j + 1] & 0xF) << 4)

    return {
        'quantized': packed_data.flatten(),
        'scales': scales,
        'metadata': {
            'original_shape': original_shape,
            'block_size': block_size,
            'num_blocks': num_blocks,
            'pad_length': pad_length,
            'nf4_levels': NF4_LEVELS
        }
    }


def decode_from_nf4(encoded_data, dtype=None):
    quantized = encoded_data['quantized']
    scales = encoded_data['scales']
    metadata = encoded_data['metadata']

    original_shape = metadata['original_shape']
    block_size = metadata['block_size']
    num_blocks = metadata['num_blocks']
    pad_length = metadata['pad_length']
    nf4_levels = metadata['nf4_levels']

    decoded_blocks = keras.ops.zeros((num_blocks, block_size), dtype=dtype)
    quantized_reshaped = keras.ops.reshape(quantized, (num_blocks, block_size // 2))

    j = keras.ops.arange(block_size // 2)
    scales_ = keras.ops.reshape(scales, (-1, 1))
    decoded_val = nf4_levels[quantized_reshaped & 0xF] * scales_
    decoded_blocks = decoded_blocks.at[:, 2 * j].set(decoded_val)
    decoded_val = nf4_levels[(quantized_reshaped >> 4) & 0xF] * scales_
    decoded_blocks = decoded_blocks.at[:, 2 * j + 1].set(decoded_val)

    # Flatten and remove padding
    decoded_flat = decoded_blocks.flatten()
    if pad_length > 0:
        decoded_flat = decoded_flat[:-pad_length]

    # Reshape to original shape
    decoded = keras.ops.reshape(decoded_flat, original_shape)
    return decoded
