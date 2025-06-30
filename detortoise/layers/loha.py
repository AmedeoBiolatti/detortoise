import keras


class LoHAWrapper(keras.layers.Layer):
    def __init__(self, base, ranks, alpha=1.0):
        super().__init__()
        self.base: keras.layers.Layer = base
        self.ranks = ranks
        self.alpha = alpha

        if self.base.built:
            self.build(None)

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

        self.kernel_loha_a = [self.add_weight(
            name=f"kernel_loha_a{i}",
            shape=shape_a + (rank,),
            initializer="he_normal"
        ) for i, rank in enumerate(self.ranks)
        ]
        self.kernel_loha_b = [self.add_weight(
            name=f"kernel_lora_b{i}",
            shape=(rank,) + shape_b,
            initializer="zeros"
        ) for i, rank in enumerate(self.ranks)
        ]

        self.built = True

    @property
    def kernel(self):
        prods = [
            (self.alpha / rank * (a @ b))
            for rank, a, b in zip(self.ranks, self.kernel_loha_a, self.kernel_loha_b)
        ]
        prod = keras.ops.prod(keras.ops.stack(prods, axis=0), axis=0)
        return self.base.kernel + prod

    def compute_output_shape(self, input_shape):
        return self.base.compute_output_shape(input_shape)

    def call(self, inputs, **kwargs):
        return type(self.base).call(self, inputs, **kwargs)
