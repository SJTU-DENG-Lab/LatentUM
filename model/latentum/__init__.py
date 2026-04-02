from .configuration_latentum import LatentUMConfig

__all__ = ["LatentUMConfig", "LatentUMModel"]


def __getattr__(name):
    if name == "LatentUMModel":
        from .modeling_latentum import LatentUMModel

        return LatentUMModel
    raise AttributeError(name)
