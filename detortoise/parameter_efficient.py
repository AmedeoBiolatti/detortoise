import keras

from detortoise.layers.lora import LoRAWrapper
from detortoise.layers.loha import LoHAWrapper
from detortoise.layers.dora import DoRAWrapper
from detortoise.patching import patch_model


def lora(
        model,
        rank,
        condition=None
):
    def patch_function(layer):
        if isinstance(layer, keras.layers.Dense | keras.layers.EinsumDense):
            if condition is None or condition(layer):
                layer = LoRAWrapper(layer, rank=rank)
        return layer

    model = patch_model(
        model,
        patch_function=patch_function
    )
    return model


def loha(
        model,
        ranks,
        condition=None
):
    def patch_function(layer):
        if isinstance(layer, keras.layers.Dense | keras.layers.EinsumDense):
            if condition is None or condition(layer):
                layer = LoHAWrapper(layer, ranks=ranks)
        return layer

    model = patch_model(
        model,
        patch_function=patch_function
    )
    return model


def dora(
        model,
        rank,
        condition=None
):
    def patch_function(layer):
        if isinstance(layer, keras.layers.Dense | keras.layers.EinsumDense):
            if condition is None or condition(layer):
                layer = DoRAWrapper(layer, rank=rank)
        return layer

    model = patch_model(
        model,
        patch_function=patch_function
    )
    return model
