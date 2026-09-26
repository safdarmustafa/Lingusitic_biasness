#!/usr/bin/env python3
"""
Class I evaluation — config-driven CLIP-family retrieval across languages.

Purpose: sanity-check that translating the SAME prompt across scripts/
languages (English / native / Hinglish) still retrieves the correct
concept's images from a pooled set of Class I (semantic-control) concepts.
Class I is expected to show *no* meaningful bias — this is the control run
before moving to Class II/III/IV.

Switch models without touching this file: edit `active_model` in
config.yaml, or run:

    python run_class1_eval.py --model openclip_xlmr
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class ImageRecord:
    concept: str
    path: Path


# ============================================================
# CONFIG + PROMPTS + POOL
# ============================================================

def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts(csv_path: Path, concepts: list[str], languages: list[str]) -> dict[str, dict[str, str]]:
    """Returns {concept: {language_column: prompt_text}}."""
    prompts: dict[str, dict[str, str]] = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        missing_cols = [lang for lang in languages if lang not in header]
        if missing_cols:
            raise SystemExit(
                f"Language column(s) {missing_cols} not found in {csv_path}. "
                f"Available columns: {header}"
            )
        for row in reader:
            if row["concept"] in concepts:
                prompts[row["concept"]] = {lang: row[lang] for lang in languages}

    missing_concepts = set(concepts) - prompts.keys()
    if missing_concepts:
        raise SystemExit(f"Concept(s) not found in {csv_path}: {sorted(missing_concepts)}")
    return prompts


def collect_pool(data_dir: Path, pool_concepts: list[str]) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for concept in pool_concepts:
        clean_dir = data_dir / concept / "clean"
        if not clean_dir.exists():
            raise SystemExit(f"Missing folder: {clean_dir}")
        for p in sorted(clean_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."):
                records.append(ImageRecord(concept=concept, path=p))
    if not records:
        raise SystemExit(f"No images found under {data_dir} for concepts: {pool_concepts}")
    return records


# ============================================================
# DEVICE + BACKEND
# ============================================================

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_backend(spec: dict, device: torch.device):
    """Returns (model, preprocess, tokenize_fn) for any registered model key."""
    backend = spec["backend"]

    if backend == "openai_clip":
        import clip

        model, preprocess = clip.load(spec["name"], device=device, jit=False)
        model.eval()

        def tokenize_fn(texts):
            return clip.tokenize(texts, truncate=True).to(device)

        return model, preprocess, tokenize_fn

    if backend == "open_clip":
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            spec["name"], pretrained=spec.get("pretrained"), device=device,
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(spec["name"])

        def tokenize_fn(texts):
            return tokenizer(texts).to(device)

        return model, preprocess, tokenize_fn

    raise SystemExit(f"Unknown backend '{backend}' — add support in load_backend().")


@torch.no_grad()
def encode_images(model, preprocess, records, device, batch_size: int) -> np.ndarray:
    feats_all = []
    for start in tqdm(range(0, len(records), batch_size), desc="images"):
        batch = records[start:start + batch_size]
        tensors = []
        for rec in batch:
            with Image.open(rec.path) as im:
                tensors.append(preprocess(im.convert("RGB")))
        pixel_batch = torch.stack(tensors).to(device)
        feats = model.encode_image(pixel_batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats_all.append(feats.float().cpu())
    return torch.cat(feats_all, dim=0).numpy()


@torch.no_grad()
def encode_texts(model, tokenize_fn, texts: list[str], device) -> np.ndarray:
    tokens = tokenize_fn(texts)
    feats = model.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.float().cpu().numpy()


# ============================================================
# RANKING
# ============================================================

def evaluate(records: list[ImageRecord], image_features: np.ndarray,
             query_concept: str, text_feature: np.ndarray, top_k: int) -> dict:
    sims = image_features @ text_feature
    order = np.argsort(-sims)[:top_k]
    hits = sum(1 for idx in order if records[idx].concept == query_concept)
    return {
        "top_k_precision": hits / top_k,
        "top_k": [
            {
                "rank": rank,
                "concept": records[idx].concept,
                "file": records[idx].path.name,
                "score": float(sims[idx]),
            }
            for rank, idx in enumerate(order, start=1)
        ],
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"), help="Path to config.yaml")
    parser.add_argument("--model", default=None, help="Override active_model from config.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    model_key = args.model or cfg["active_model"]
    if model_key not in cfg["models"]:
        raise SystemExit(f"Model '{model_key}' not in config.models. Options: {list(cfg['models'])}")
    spec = cfg["models"][model_key]

    data_dir = ROOT / cfg["data_dir"]
    results_dir = ROOT / cfg["results_dir"] / model_key
    results_dir.mkdir(parents=True, exist_ok=True)

    query_concepts = cfg["concepts"]
    pool_concepts = cfg.get("pool_concepts") or sorted(
        p.name for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    languages = cfg["languages"]
    top_k = cfg.get("top_k", 10)
    prompt_variants = cfg["prompt_variants"]

    records = collect_pool(data_dir, pool_concepts)

    print(f"Model         : {spec['display']}  (key: {model_key})")
    print(f"Pool          : {len(records)} images across {len(pool_concepts)} concept(s)")
    print(f"Query concepts: {query_concepts}")
    print(f"Languages     : {languages}")
    print(f"Prompt variants: {list(prompt_variants)}")

    device = get_device()
    print(f"Device        : {device}")

    model, preprocess, tokenize_fn = load_backend(spec, device)
    image_features = encode_images(model, preprocess, records, device, cfg.get("batch_size", 16))

    all_summary_rows = []
    for variant_name, csv_name in prompt_variants.items():
        prompts = load_prompts(ROOT / csv_name, query_concepts, languages)
        variant_dir = results_dir / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)

        for concept in query_concepts:
            lang_texts = prompts[concept]
            text_features = encode_texts(model, tokenize_fn, list(lang_texts.values()), device)

            for lang, text_feature in zip(lang_texts.keys(), text_features):
                result = evaluate(records, image_features, concept, text_feature, top_k)
                print(f"  [{variant_name} / {concept} / {lang}] top-{top_k} precision = "
                      f"{result['top_k_precision']:.2f}   prompt='{lang_texts[lang]}'")

                out_json = variant_dir / f"{concept}_{lang}.json"
                out_json.write_text(
                    json.dumps(
                        {
                            "model": spec["display"],
                            "variant": variant_name,
                            "concept": concept,
                            "language": lang,
                            "prompt": lang_texts[lang],
                            **result,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                all_summary_rows.append({
                    "variant": variant_name,
                    "concept": concept,
                    "language": lang,
                    "prompt": lang_texts[lang],
                    "top_k_precision": result["top_k_precision"],
                })

    summary_csv = results_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["variant", "concept", "language", "prompt", "top_k_precision"]
        )
        writer.writeheader()
        writer.writerows(all_summary_rows)

    print(f"\nDone. Results -> {results_dir}  (compare variant subfolders + summary.csv)")


if __name__ == "__main__":
    main()
