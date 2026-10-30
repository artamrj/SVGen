<div align="center">

# SVGen

**Text in, clean SVG icon out.**

[![tests](https://github.com/artamrj/SVGen/actions/workflows/tests.yml/badge.svg)](https://github.com/artamrj/SVGen/actions/workflows/tests.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Apple Silicon](https://img.shields.io/badge/runs%20on-Apple%20Silicon-black.svg)

A small language model fine-tuned to draw black-and-white line icons as real, editable SVG code. Trained and run on a MacBook.

![SVGen samples](docs/showcase.png)

</div>

```bash
make svg PROMPT="shopping cart" OUT=cart.svg
```

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M4 19A2 2 0 1 0 8 19A2 2 0 1 0 4 19"/><path d="M15 19A2 2 0 1 0 19 19A2 2 0 1 0 15 19"/>
  <path d="M17 17H6V3H4"/><path d="M6 5L20 6L19 13H6"/>
</svg>
```

## How it works

```mermaid
flowchart LR
    P["Prompt<br/>'shopping cart'"] --> M["Qwen2.5-Coder-0.5B<br/>+ SVGen LoRA (MLX)"]
    M -->|"4 candidates"| V{"Strict SVG<br/>validator"}
    V -->|"invalid"| X["dropped / salvaged"]
    V -->|"valid"| R["Render<br/>(librsvg)"]
    R --> S["SigLIP2 scores<br/>image vs. text"]
    S -->|"best match"| W["Safe svg wrapper"]
    W --> O["icon.svg"]
```

The model writes **only shapes** (`path`, `circle`, `rect`, `line`, `ellipse`). SVGen adds the `<svg>` wrapper, so every output is well-formed, safe (no scripts or external links), and recolorable with CSS `color`.

## Training

```mermaid
flowchart TD
    subgraph Data
        A["Lucide · Tabler · Iconoir · Heroicons<br/>(npm, MIT/ISC)"] --> B["Clean + normalize<br/>absolute paths, 0.1 precision"]
        B --> C["Prompts from names + tags<br/>'line icon of a lock'"]
        C --> D["Split by concept<br/>train / valid / test"]
    end
    subgraph FT["LoRA fine-tune (mlx_lm)"]
        D --> S1["Stage 1 · 2000 steps"]
        S1 --> S2["Stage 2 · 3000 steps"]
        S2 --> S3["Stage 3 · 1200 steps<br/>rebalanced: base icons > variants"]
    end
    S3 --> E["Release fp16 adapter<br/>models/svgen-0.3-mono (35 MB)"]
    E --> F["Evaluate<br/>validity + SigLIP2 match"]
```

| | |
|---|---|
| **Base model** | Qwen2.5-Coder-0.5B-Instruct |
| **Method** | LoRA, rank 32, all layers (17.6M trainable params) |
| **Data** | 8,873 unique 24×24 line icons, ~24k prompt–SVG pairs |
| **Hardware** | MacBook Pro M1 Pro, 16 GB, Apple MLX |
| **Training time** | ~5 h at low priority, laptop stayed usable |

## Results

Measured on held-out concepts the model never saw in training:

| Metric | Score |
|---|---|
| Valid SVG samples | **98.8%** |
| Prompts with a usable icon | **100%** |
| SigLIP2 text match | **0.165** (real icons: 0.184) |

Common UI concepts work well. Icons that are one long precise shape (star, rocket, bicycle) and unusual prompts are still weak. Full details are in the [model card](docs/model_card_svgen_0.3.md).

## Quickstart

Requires an Apple Silicon Mac.

```bash
make setup               # create venv, install dependencies
brew install librsvg     # rendering for reranking
make svg PROMPT="lock" OUT=lock.svg
```

```python
from svgen.generate import text_to_svg
svg = text_to_svg("cloud", n=4)
```

Train from scratch on a Mac: `make data-download data train release eval`.

Train the larger 1.5B model on a free Colab GPU:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/artamrj/SVGen/blob/main/notebooks/train_colab.ipynb)

## Project structure

```mermaid
flowchart LR
    subgraph PKG["svgen/"]
        svg["svg.py<br/>format · validator"]
        data["data.py<br/>dataset builder"]
        prompting["prompting.py<br/>chat format"]
        gen["generate.py<br/>sampling + ranking"]
        rerank["rerank.py<br/>SigLIP2"]
        evaluate["evaluate.py"]
        release["release.py"]
    end
    configs["configs/lora_mono*.yaml"] --> mlx["mlx_lm lora"]
    data --> prompting
    data --> svg
    data -->|"data/processed"| mlx
    mlx -->|"checkpoints/"| release
    release -->|"models/"| gen
    gen --> prompting
    gen --> svg
    gen --> rerank
    evaluate --> gen
```

## Roadmap

- [x] Black-and-white line icons on a laptop (v0.3)
- [ ] Larger base model (1.5B) trained on a cloud GPU
- [ ] Visual captions from a vision model, instead of names and tags
- [ ] Reinforcement learning from rendered output (GRPO + SigLIP2 reward)
- [ ] Filled icons, then color
- [ ] Rust runtime inference (the safety validator already exists in `runtime/`)

## License

Code: [MIT](LICENSE). The model adapter builds on Qwen2.5-Coder (Apache-2.0), and training icons keep their own licenses (below).

## Credits

Icons for training: [Lucide](https://lucide.dev) (ISC), [Tabler Icons](https://tabler.io/icons), [Iconoir](https://iconoir.com), and [Heroicons](https://heroicons.com) (MIT). Base model: [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct).
