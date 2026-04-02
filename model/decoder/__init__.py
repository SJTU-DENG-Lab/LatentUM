from .configuration_decoder import LatentUMDecoderConfig

__all__ = ["LatentUMDecoderConfig", "LatentUMDecoderModel", "LatentUMRefDecoderModel"]


def __getattr__(name):
    if name == "LatentUMDecoderModel":
        from .modeling_decoder import LatentUMDecoderModel

        return LatentUMDecoderModel
    if name == "LatentUMRefDecoderModel":
        from .modeling_decoder import LatentUMRefDecoderModel

        return LatentUMRefDecoderModel
    raise AttributeError(name)
