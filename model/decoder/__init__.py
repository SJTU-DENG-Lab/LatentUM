from .configuration_decoder import LatentUMDecoderConfig

__all__ = ["LatentUMDecoderConfig", "LatentUMDecoderModel"]


def __getattr__(name):
    if name == "LatentUMDecoderModel":
        from .modeling_decoder import LatentUMDecoderModel

        return LatentUMDecoderModel
    raise AttributeError(name)
