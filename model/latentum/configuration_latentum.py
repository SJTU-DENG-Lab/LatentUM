import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def resolve_pretrained_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.exists():
        return resolved
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=str(path)))


@dataclass
class LatentUMConfig:
    base_model_name_or_path: str | None = None
    quantizer_ckpt_path: str | None = None
    llm_hidden_size: int = 2560
    mixture_mode: str = "mot"
    embedding_dim: int = 256
    image_size: int = 448
    num_image_tokens: int = 256
    max_num_patches: int = 12
    image_token: str = "<image>"
    internvl_config: dict[str, Any] = field(default_factory=dict)
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
        resolved_path = resolve_pretrained_path(path)
        with open(resolved_path / "config.json", "r", encoding="utf-8") as f:
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
        resolved_path = resolve_pretrained_path(path)
        with open(resolved_path / "config.json", "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
