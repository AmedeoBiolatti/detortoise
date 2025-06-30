import keras


class LoRAWrapper(keras.layers.Layer):
    def __init__(self, base, rank, diag=False, alpha=1.0, trainable=True):
        super().__init__()
        self.base: keras.layers.Layer = base
        self.rank = rank
        self.alpha = alpha
        self.diag = diag
        self.trainable = trainable

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

        self.kernel_lora_a = self.add_weight(
            name="kernel_lora_a",
            shape=shape_a + (self.rank,),
            initializer="he_normal"
        )
        self.kernel_lora_b = self.add_weight(
            name="kernel_lora_b",
            shape=(self.rank,) + shape_b,
            initializer="he_normal" if self.diag else "zeros"
        )
        if self.diag:
            self.kernel_lora_d = self.add_weight(
                name="kernel_lora_d",
                shape=tuple([1] * len(shape_a)) + (self.rank,),
                initializer="zeros"
            )

        self.built = True

    @property
    def kernel(self):
        mult = self.alpha / self.rank
        a = self.kernel_lora_a
        b = self.kernel_lora_b
        if self.diag:
            a = a * self.kernel_lora_d
        return self.base.kernel + mult * (a @ b)

    def call(self, inputs, **kwargs):
        return type(self.base).call(self, inputs, **kwargs)
