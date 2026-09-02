#!/usr/bin/env python3
"""
Path-blind CLIP / OpenCLIP retrieval for linguistic bias.

Protocol
--------
CLIP never enters culture subfolders.  Photos are copied onto the Dolls/
floor with random names, shuffled, and the model only reads those files.
It never opens Indian/, European/, African/, or Japanese/.  A private
map kept outside Dolls/ is used only after ranking.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path.home() / "Desktop" / "Cultural_CLIP_Pilot"
DOLLS_DIR = BASE_DIR / "Dolls"
RESULTS_DIR = BASE_DIR / "results"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SKIP_DIR_NAMES = {".", "..", "__pycache__"}

TOP_K = 10
BATCH_SIZE = 16
SEED = 42

# Neutral prompts.  Culture / region words are intentionally absent.
PROMPTS = {
    "english": "image of a doll",
    "hindi": "गुड़िया की तस्वीर",
}

# One folder per model.  Inside: english/01.jpg … 10.jpg and hindi/01.jpg … 10.jpg.
MODELS = [
    {
        "key": "clip_vitb32",
        "folder": "openclip_vit32",
        "backend": "openai_clip",
        "name": "ViT-B/32",
        "pretrained": None,
        "display": "CLIP ViT-B/32",
    },
    {
        "key": "openclip_vitb32_laion2b",
        "folder": "openclip_vit32_laion2b",
        "backend": "open_clip",
        "name": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
        "display": "OpenCLIP ViT-B/32 LAION-2B",
    },
    {
        "key": "openclip_xlmr",
        "folder": "openclip_xlmr",
        "backend": "open_clip",
        "name": "xlm-roberta-base-ViT-B-32",
        "pretrained": "laion5b_s13b_b90k",
        "display": "OpenCLIP XLM-R ViT-B/32",
    },
]


# ============================================================
# IMAGE POOL  (path-blind)
# ============================================================

@dataclass
class ImageRecord:
    """One doll image.

    `anon_id` is what we print and save.  `path` is used only to load
    pixels.  `culture` / `original_filename` are analysis-only metadata
    and are never encoded.
    """

    anon_id: str
    path: Path
    culture: str
    original_filename: str


def source_images_in_subfolders(dolls_dir: Path) -> list[Path]:
    """Originals live one level down (Indian/, European/, ...). CLIP never reads these."""
    paths = []
    for child in sorted(dolls_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        for p in sorted(child.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."):
                paths.append(p)
    return paths


def rebuild_flat_pool(dolls_dir: Path) -> list[dict]:
    """Copy every doll onto the Dolls/ floor with a random name, shuffled.

    After this, CLIP only sees Dolls/*.jpg — never a culture folder.
    """
    sources = source_images_in_subfolders(dolls_dir)
    if not sources:
        raise SystemExit(f"No images inside culture subfolders of {dolls_dir}")

    for p in dolls_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            p.unlink()

    rng = random.Random(SEED)
    order = sources[:]
    rng.shuffle(order)

    mapping = []
    used: set[str] = set()
    for src in order:
        while True:
            name = rng.randbytes(4).hex() + src.suffix.lower()
            if name not in used:
                used.add(name)
                break
        shutil.copy2(src, dolls_dir / name)
        mapping.append(
            {
                "flat_name": name,
                "culture": src.parent.name,
                "original_filename": src.name,
            }
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    map_path = RESULTS_DIR / "pool_map.csv"
    with map_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["flat_name", "culture", "original_filename"]
        )
        writer.writeheader()
        writer.writerows(mapping)
    return mapping


def collect_image_pool(dolls_dir: Path) -> list[ImageRecord]:
    """Read only files sitting directly in Dolls/. Never descend into subfolders."""
    meta_by_name = {}
    map_path = RESULTS_DIR / "pool_map.csv"
    if map_path.exists():
        with map_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                meta_by_name[row["flat_name"]] = row

    paths = [
        p
        for p in dolls_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and not p.name.startswith(".")
    ]
    rng = random.Random(SEED)
    rng.shuffle(paths)

    records = []
    for i, path in enumerate(paths):
        meta = meta_by_name.get(path.name, {})
        records.append(
            ImageRecord(
                anon_id=f"img_{i:03d}",
                path=path,
                culture=meta.get("culture", ""),
                original_filename=meta.get("original_filename", ""),
            )
        )
    return records


def pick_devanagari_font():
    """Use a system font that can render Hindi in matplotlib titles."""
    preferred = [
        "Kohinoor Devanagari",
        "Devanagari Sangam MN",
        "ITF Devanagari",
        "Nirmala UI",
        "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.family"] = name
            return name
    return None


# ============================================================
# DEVICE + ENCODING
# ============================================================

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_backend(spec: dict, device: torch.device):
    """Return (model, preprocess, tokenize_fn) for a CLIP or OpenCLIP spec."""
    if spec["backend"] == "openai_clip":
        import clip

        model, preprocess = clip.load(spec["name"], device=device, jit=False)
        model.eval()

        def tokenize_fn(texts):
            return clip.tokenize(texts, truncate=True).to(device)

        return model, preprocess, tokenize_fn, "openai_clip"

    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        spec["name"],
        pretrained=spec["pretrained"],
        device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(spec["name"])

    def tokenize_fn(texts):
        return tokenizer(texts).to(device)

    return model, preprocess, tokenize_fn, "open_clip"


@torch.no_grad()
def encode_images(model, preprocess, records, device, backend: str) -> np.ndarray:
    """Encode pixels only.  Paths / names never enter the model."""
    features = []
    for start in tqdm(range(0, len(records), BATCH_SIZE), desc="  images", leave=False):
        batch = records[start : start + BATCH_SIZE]
        tensors = []
        for rec in batch:
            with Image.open(rec.path) as im:
                tensors.append(preprocess(im.convert("RGB")))
        pixel_batch = torch.stack(tensors).to(device)
        feats = model.encode_image(pixel_batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        features.append(feats.float().cpu())
    return torch.cat(features, dim=0).numpy()


@torch.no_grad()
def encode_texts(model, tokenize_fn, texts, device) -> np.ndarray:
    tokens = tokenize_fn(texts)
    feats = model.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.float().cpu().numpy()


# ============================================================
# RANKING + ANALYSIS
# ============================================================

def rank_against_prompt(image_features: np.ndarray, text_feature: np.ndarray) -> np.ndarray:
    """Cosine similarity because features are already L2-normalized."""
    return image_features @ text_feature.squeeze()


def culture_summary(records, similarities, top_k: int) -> dict:
    cultures = [r.culture for r in records]
    prior = Counter(cultures)
    n = len(records)

    order = np.argsort(-similarities)
    top = [records[i] for i in order[:top_k]]
    top_counts = Counter(r.culture for r in top)

    per_culture_sims = defaultdict(list)
    for rec, sim in zip(records, similarities):
        per_culture_sims[rec.culture].append(float(sim))

    rows = []
    for culture in sorted(prior):
        mean_sim = float(np.mean(per_culture_sims[culture]))
        share_pool = prior[culture] / n
        share_top = top_counts.get(culture, 0) / top_k
        lift = (share_top / share_pool) if share_pool else 0.0
        rows.append(
            {
                "culture": culture,
                "n_in_pool": prior[culture],
                "pool_share": round(share_pool, 4),
                "top_k_count": top_counts.get(culture, 0),
                "top_k_share": round(share_top, 4),
                "lift_vs_pool": round(lift, 3),
                "mean_similarity": round(mean_sim, 4),
            }
        )
    return {
        "prior": dict(prior),
        "top_k_counts": dict(top_counts),
        "rows": rows,
        "order": order,
    }


def save_rankings_csv(path: Path, records, similarities, culture_of):
    order = np.argsort(-similarities)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "anon_id", "similarity", "culture", "filename"],
        )
        writer.writeheader()
        for rank, idx in enumerate(order, start=1):
            rec = records[idx]
            writer.writerow(
                {
                    "rank": rank,
                    "anon_id": rec.anon_id,
                    "similarity": f"{similarities[idx]:.6f}",
                    "culture": rec.culture,
                    "filename": rec.path.name,
                }
            )


# ============================================================
# FIGURES
# ============================================================

def _thumb(path: Path, size=160) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size))
        canvas = Image.new("RGB", (size, size), (240, 240, 240))
        canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
        return np.asarray(canvas)


def save_ranked_results(
    records,
    similarities,
    model_dir: Path,
    lang: str,
    prompt: str,
    model_name: str,
    top_k: int,
):
    """Write top-k as separate images plus one CSV of CLIP scores for this prompt."""
    lang_dir = model_dir / lang
    if lang_dir.exists():
        shutil.rmtree(lang_dir)
    lang_dir.mkdir(parents=True)

    order = np.argsort(-similarities)[:top_k]
    rows = []
    for rank, idx in enumerate(order, start=1):
        rec = records[idx]
        result_name = f"{rank:02d}{rec.path.suffix.lower()}"
        shutil.copy2(rec.path, lang_dir / result_name)
        rows.append(
            {
                "rank": rank,
                "result_file": f"{lang}/{result_name}",
                "clip_score": f"{float(similarities[idx]):.6f}",
                "prompt": prompt,
                "model": model_name,
                "pool_file": rec.path.name,
                "culture": rec.culture,
                "original_filename": rec.original_filename,
            }
        )

    csv_path = model_dir / f"{lang}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "result_file",
                "clip_score",
                "prompt",
                "model",
                "pool_file",
                "culture",
                "original_filename",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return lang_dir, csv_path


def plot_bias_summary(all_summaries: list, out_path: Path, top_k: int):
    """Grouped bars: mean similarity per culture, one panel per model."""
    models = []
    for item in all_summaries:
        if item["display"] not in models:
            models.append(item["display"])
    cultures = sorted({row["culture"] for item in all_summaries for row in item["rows"]})
    languages = []
    for item in all_summaries:
        if item["language"] not in languages:
            languages.append(item["language"])

    n_models = len(models)
    fig, axes = plt.subplots(n_models, 2, figsize=(11, 3.4 * n_models), squeeze=False)

    x = np.arange(len(cultures))
    width = 0.35
    colors = {"english": "#3b6d99", "hindi": "#b55a3c"}

    for m_i, model_name in enumerate(models):
        # mean similarity
        ax = axes[m_i][0]
        for l_i, lang in enumerate(languages):
            match = next(
                s for s in all_summaries if s["display"] == model_name and s["language"] == lang
            )
            means = []
            for c in cultures:
                row = next(r for r in match["rows"] if r["culture"] == c)
                means.append(row["mean_similarity"])
            ax.bar(
                x + (l_i - 0.5) * width,
                means,
                width,
                label=lang,
                color=colors.get(lang, None),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(cultures)
        ax.set_ylabel("Mean cosine similarity")
        ax.set_title(f"{model_name}\nmean similarity by culture")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # top-k counts
        ax = axes[m_i][1]
        for l_i, lang in enumerate(languages):
            match = next(
                s for s in all_summaries if s["display"] == model_name and s["language"] == lang
            )
            counts = []
            for c in cultures:
                row = next(r for r in match["rows"] if r["culture"] == c)
                counts.append(row["top_k_count"])
            ax.bar(
                x + (l_i - 0.5) * width,
                counts,
                width,
                label=lang,
                color=colors.get(lang, None),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(cultures)
        ax.set_ylabel(f"Count in top {top_k}")
        ax.set_title(f"{model_name}\ntop-{top_k} retrievals by culture")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Path-blind retrieval: English vs Hindi prompt  ·  culture labels added after ranking",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def reset_results_dir():
    """Wipe previous run. Each model keeps english/ and hindi/ with separate ranked files."""
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir()


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    reset_results_dir()

    rebuild_flat_pool(DOLLS_DIR)
    records = collect_image_pool(DOLLS_DIR)
    if not records:
        raise SystemExit(f"No images found under {DOLLS_DIR}")

    print(f"Flat pool: {len(records)} images on the Dolls/ floor (no subfolders).")
    print(f"English prompt: {PROMPTS['english']}")
    print(f"Hindi prompt:   {PROMPTS['hindi']}")

    device = get_device()
    print(f"Device: {device}")

    for spec in MODELS:
        print(f"Loading {spec['display']}")
        model, preprocess, tokenize_fn, backend = load_backend(spec, device)
        image_features = encode_images(model, preprocess, records, device, backend)
        text_features = encode_texts(
            model, tokenize_fn, list(PROMPTS.values()), device
        )

        model_dir = RESULTS_DIR / spec["folder"]
        model_dir.mkdir(parents=True, exist_ok=True)

        for lang_i, (lang, prompt) in enumerate(PROMPTS.items()):
            sims = rank_against_prompt(image_features, text_features[lang_i])
            lang_dir, csv_path = save_ranked_results(
                records,
                sims,
                model_dir,
                lang,
                prompt,
                spec["display"],
                TOP_K,
            )
            print(f"  {lang}: {TOP_K} images -> {lang_dir}")
            print(f"  {lang}: scores -> {csv_path}")

        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    print("Done.")


if __name__ == "__main__":
    main()
