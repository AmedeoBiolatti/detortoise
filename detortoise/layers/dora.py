import keras


class DoRAWrapper(keras.layers.Layer):
    def __init__(self, base, rank, alpha=1.0):
        super().__init__()
        self.base: keras.layers.Layer = base
        self.rank = rank
        self.alpha = alpha

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        if item == "base":
            raise AttributeError
        return getattr(self.base, item)

    def build(self, input_shape):
        if self.built:
            return
        if not self.base.built:
            self.base.build(input_shape)
        assert hasattr(self.base, "kernel")
        self.base.kernel.trainable = False

        shape_a = self.base.kernel.shape[:-1]
        shape_b = self.base.kernel.shape[-1:]

        self.kernel_dora_a = self.add_weight(
            name="kernel_dora_a",
            shape=shape_a + (self.rank,),
            initializer="he_normal"
        )
        self.kernel_dora_b = self.add_weight(
            name="kernel_dora_b",
            shape=(self.rank,) + shape_b,
            initializer="zeros"
        )

        kernel_flat = keras.ops.reshape(self.base.kernel, (-1,) + shape_b)
        kernel_norm = keras.ops.norm(kernel_flat, axis=0)
        self.kernel_dora_m = self.add_weight(
            name="kernel_dora_m",
            shape=shape_b,
            initializer=keras.initializers.Constant(kernel_norm)
        )

        self.built = True

    @property
    def kernel(self):
        mult = self.alpha / self.rank
        a = self.kernel_dora_a
        b = self.kernel_dora_b

        kernel = self.base.kernel + (a @ b) * mult
        kernel_flat = keras.ops.reshape(kernel, (-1, kernel.shape[-1]))
        kernel_norm = keras.ops.norm(kernel_flat, axis=0)

        kernel = kernel * (self.kernel_dora_m / kernel_norm)
        return kernel

    def call(self, inputs, **kwargs):
        return type(self.base).call(self, inputs, **kwargs)
