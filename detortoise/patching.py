import keras


def patch_model(
        model,
        patch_function
):
    patch_memory = {}

    def patching_function(layer):
        if hasattr(layer, "_tracker"):
            layer._tracker.unlock()
        for name in layer.__dir__():
            if hasattr(type(layer), name) and isinstance(getattr(type(layer), name), property):
                continue
            a = a_original = getattr(layer, name)
            if isinstance(a, keras.layers.Layer):
                a = patching_function(a)
                if hasattr(layer, "_tracker"):
                    # remove tracking of original sub-layer
                    layer._tracker.untrack(a_original)
                setattr(layer, name, a)
        if hasattr(layer, "_tracker"):
            layer._tracker.lock()

        # update layer
        if layer in patch_memory:
            patched_layer = patch_memory[layer]
        else:
            patched_layer = patch_function(layer)
            patch_memory[layer] = patched_layer
        layer = patched_layer



        return layer

    model = keras.models.clone_model(
        model,
        clone_function=patching_function,
        recursive=True
    )
    return model
