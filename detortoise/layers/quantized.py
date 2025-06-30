import keras
from absl import logging

from detortoise.quantization import nf4

"""
TODO: check necessary fields for
- [x] Dense
- [x] EinsumDense
- [ ] Conv
"""
fields = [
    "activation",
    "equation",
    "units"
]


class NF4(keras.layers.Layer):
    def __init__(self, base, activity_regularizer=None, **kwargs):
        if not base.built:
            logging.warning(f"Using a {self.__class__.__name__} layer of an unbuilt base layer "
                            "could not return the expected results.")
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.base_type = type(base)

        try:
            encoding = nf4.encode_to_nf4(base.kernel.value)
        except AttributeError:
            encoding = nf4.encode_to_nf4(base.kernel)

        self.kernel_quantized = self.add_weight(
            shape=encoding['quantized'].shape,
            name="kernel_quantized",
            initializer=keras.initializers.constant(encoding['quantized']),
            trainable=False,
            dtype="uint8"
        )
        self.kernel_scales = self.add_weight(
            shape=encoding['scales'].shape,
            name="kernel_scales",
            initializer=keras.initializers.constant(encoding['scales']),
            trainable=False
        )
        self.original_shape = encoding["metadata"]["original_shape"]
        self.pad_length = encoding["metadata"]["pad_length"]
        self.block_size = encoding["metadata"]["block_size"]
        self.num_blocks = encoding["metadata"]["num_blocks"]

        if hasattr(base, "bias"):
            if base.bias is not None:
                self.bias = self.add_weight(
                    shape=base.bias.shape,
                    name="bias",
                    initializer=keras.initializers.constant(base.bias),
                )
            else:
                self.bias = None
        for f in fields:
            if hasattr(base, f):
                setattr(self, f, getattr(base, f))

    @property
    def kernel(self):
        if keras.backend.backend() == "jax":
            from detortoise.quantization.kernels.nf4_jax import dequantize
            return dequantize(
                self.kernel_quantized.value,
                self.kernel_scales.value,
                self.original_shape,
                self.num_blocks,
                self.pad_length,
                nf4.NF4_LEVELS,
                #dtype=self.dtype_policy.dtype
            )
        kernel = nf4.decode_from_nf4({
            "quantized": self.kernel_quantized,
            "scales": self.kernel_scales,
            "metadata": {
                "original_shape": self.original_shape,
                "block_size": self.block_size,
                "num_blocks": self.num_blocks,
                "pad_length": self.pad_length,
                "nf4_levels": nf4.NF4_LEVELS
            }
        })
        return kernel

    def compute_output_shape(self, input_shape):
        return self.base_type.compute_output_shape(self, input_shape)

    def compute_output_spec(self, *args, **kwargs):
        return self.base_type.compute_output_spec(self, *args, **kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, *args, **kwargs):
        out = self.base_type.call(self, inputs, *args, **kwargs)
        return keras.ops.cast(out, self.dtype_policy.compute_dtype)
