import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LatentUMConfig:
    base_model_name_or_path: str
    quantizer_ckpt_path: str | None = None
    llm_hidden_size: int = 2560
    mixture_mode: str = "mot"
    embedding_dim: int = 256
    image_size: int = 448
    num_image_tokens: int = 256
    max_num_patches: int = 12
    image_token: str = "<image>"
    model: dict[str, Any] = field(default_factory=dict)
    quantizer: dict[str, Any] = field(default_factory=dict)
    head: dict[str, Any] = field(default_factory=dict)
    legacy_checkpoint_path: str | None = None
    legacy_state_key: str = "module"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LatentUMConfig":
        return cls(**data)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "LatentUMConfig":
        with open(Path(path) / "config.json", "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


@dataclass
class LatentUMDecoderConfig:
    sd3_5_path: str
    context_dim: int = 256
    load_pretrained: bool = False
    image_size: int = 448
    ref_mode: str = "none"
    num_ref_frames: int = 0
    legacy_checkpoint_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LatentUMDecoderConfig":
        return cls(**data)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "LatentUMDecoderConfig":
        with open(Path(path) / "config.json", "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
