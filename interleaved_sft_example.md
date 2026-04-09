# Interleaved SFT Example

This repo includes a minimal example showing how to fine-tune `LatentUM` on interleaved text-image supervision.

The example keeps the public dataset format generic, but the shipped sample follows the real FrozenLake task wording and action/image ordering used in the original training setup.

## Data Format

Use one JSONL object per sample:

```json
{
  "id": "sample-001",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Task prompt..."},
        {"type": "image", "image": "images/0.png"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "R"},
        {"type": "image", "image": "images/1.png"},
        {"type": "text", "text": "D"},
        {"type": "image", "image": "images/2.png"},
        {"type": "text", "text": "Done!"}
      ]
    }
  ]
}
```

Rules in this example:

- `messages` must be exactly `[user, assistant]`
- v1 supports exactly one user image
- assistant content may interleave multiple text and image items
- assistant content must end with text
- image paths are relative to the JSONL file unless absolute

Internally, each image item is expanded into LatentUM's placeholder token block:

```text
<img><IMG_CONTEXT>...256 times...</img>
```

The user image becomes the understanding image, and assistant images become generation targets.

## Shipped FrozenLake Sample

The official sample lives in:

- `asset/frozenlake_interleaved_example/sample.jsonl`
- `asset/frozenlake_interleaved_example/images/`

It is copied from a real FrozenLake training sample and contains 4 target images:

- `0.png`: initial maze state
- `1.png` to `4.png`: target observations after actions `R`, `R`, `R`, `D`

## Training Scripts

The example is intentionally split into two scripts because `LatentUM` decouples text understanding and visual generation. The understanding branch is not affected by the generation branch, but the generation branch depends on the understanding branch outputs, so the recommended recipe is to first adapt understanding and then adapt generation.

Language-path SFT:

```bash
uv run python script/train_interleaved_lang_only.py \
  --model-path ckpt/latentum-base \
  --data asset/frozenlake_interleaved_example/sample.jsonl \
  --output-dir outputs/interleaved-lang \
  --batch-size 1 \
  --max-seq-length 1536 \
  --num-steps 10
```

Vision-path SFT:

```bash
uv run python script/train_interleaved_vision_only.py \
  --model-path outputs/interleaved-lang/checkpoint-10 \
  --data asset/frozenlake_interleaved_example/sample.jsonl \
  --output-dir outputs/interleaved-vision \
  --batch-size 1 \
  --max-seq-length 1536 \
  --num-steps 10
```

Recommended order:

1. Train `lang_only` first.
2. Train `vision_only` second.

## What Each Script Trains

`train_interleaved_lang_only.py`

- freezes the vision path and AR image head
- injects the user image into the first image block
- injects assistant target image latents into later image blocks
- optimizes text cross-entropy on assistant text only

`train_interleaved_vision_only.py`

- freezes the language path except `_vision` parameters
- unfreezes `ar_head`
- appends dedicated generation blocks for assistant images
- optimizes visual autoregressive token prediction