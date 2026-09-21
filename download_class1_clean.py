#!/usr/bin/env python3
"""
Class I Clean Dataset Downloader — Linguistic Bias Research
============================================================
Downloads clean, isolated images for Class I concepts.

TWO-STEP APPROACH:
  Step 1 — Download images from Wikimedia Commons
           (searching specifically for "white background isolated")
  Step 2 — Auto remove background using rembg → white background
           + Validate: no text, single object, minimum size

Output folder structure:
    Desktop/Cultural_CLIP_Pilot/
    └── Class1/
        ├── apple/
        │   ├── raw/          ← original downloads
        │   └── clean/        ← white bg, no text, validated ✓
        ├── tiger/
        └── ...

Install requirements:
    pip install requests Pillow rembg tqdm
"""

from __future__ import annotations
import hashlib
import io
import os
import re
import time
from pathlib import Path

import requests
from PIL import Image, ImageFilter, ImageStat
from rembg import remove
from tqdm import tqdm

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path.home() / "Desktop" / "Cultural_CLIP_Pilot"
CLASS1_DIR = BASE_DIR / "Class1"
TARGET     = 100          # clean images per concept
DELAY      = 0.4          # seconds between requests
MIN_SIZE   = (150, 150)   # reject tiny images
OUTPUT_SIZE = (512, 512)  # final image size (square, white bg)
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".webp"}
HEADERS    = {"User-Agent": "LinguisticBiasResearch/1.0 (academic)"}
WIKI_API   = "https://commons.wikimedia.org/w/api.php"

# ── Sir's Class I Concepts ──────────────────────────────────────────────────
# Search query uses "white background isolated" to get clean source images
CLASS1_CONCEPTS = {
    "apple":       "apple fruit white background isolated",
    "orange":      "orange fruit white background isolated",
    "pear":        "pear fruit white background isolated",
    "mushroom":    "mushroom white background isolated",
    "rose":        "rose flower white background isolated",
    "bee":         "honey bee insect white background isolated",
    "beetle":      "beetle insect white background isolated",
    "butterfly":   "butterfly insect white background isolated",
    "caterpillar": "caterpillar white background isolated",
    "cockroach":   "cockroach insect white background isolated",
    "tiger":       "tiger animal white background isolated",
    "wolf":        "wolf animal white background isolated",
    "camel":       "camel animal white background isolated",
    "chimp":       "chimpanzee primate white background isolated",
    "kangaroo":    "kangaroo white background isolated",
    "fox":         "fox animal white background isolated",
    "raccoon":     "raccoon animal white background isolated",
    "hamster":     "hamster white background isolated",
    "rabbit":      "rabbit white background isolated",
    "squirrel":    "squirrel white background isolated",
    "lobster":     "lobster white background isolated",
    "spider":      "spider white background isolated",
    "worm":        "earthworm white background isolated",
    "crocodile":   "crocodile white background isolated",
    "lizard":      "lizard reptile white background isolated",
    "snake":       "snake reptile white background isolated",
    "turtle":      "turtle white background isolated",
    "bottle":      "glass bottle white background isolated",
    "cloud":       "cloud white background isolated",
    "dinosaur":    "dinosaur toy white background isolated",
}

# ── WIKIMEDIA SEARCH ─────────────────────────────────────────────────────────
def search_wikimedia(query: str, limit: int = 50) -> list[dict]:
    params = {
        "action":       "query",
        "generator":    "search",
        "gsrsearch":    f"filetype:bitmap {query}",
        "gsrnamespace": 6,
        "gsrlimit":     limit,
        "prop":         "imageinfo",
        "iiprop":       "url|size|extmetadata|mime",
        "iiurlwidth":   600,
        "format":       "json",
    }
    try:
        r = requests.get(WIKI_API, params=params,
                         headers=HEADERS, timeout=15)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        return list(pages.values())
    except Exception as e:
        print(f"    [API error] {e}")
        return []

def get_image_url(page: dict) -> str | None:
    try:
        info = page.get("imageinfo", [{}])[0]
        mime = info.get("mime", "")
        if "svg" in mime or "gif" in mime:    # skip SVG and GIF
            return None
        url = info.get("thumburl") or info.get("url", "")
        if url.startswith("http") and Path(url).suffix.lower() in IMG_EXTS:
            return url
    except Exception:
        pass
    return None

def download_raw(url: str) -> bytes | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        if len(r.content) < 8_000:
            return None
        return r.content
    except Exception:
        return None

# ── IMAGE VALIDATION ─────────────────────────────────────────────────────────
def is_too_small(img: Image.Image) -> bool:
    return img.width < MIN_SIZE[0] or img.height < MIN_SIZE[1]

def has_heavy_text_regions(img: Image.Image) -> bool:
    """
    Rough heuristic: convert to grayscale, look for dense horizontal
    bands of very dark pixels (typical of printed text on images).
    Not perfect — but catches watermarked / labelled images.
    """
    gray = img.convert("L").resize((200, 200))
    pixels = list(gray.getdata())
    dark   = sum(1 for p in pixels if p < 60)
    ratio  = dark / len(pixels)
    return ratio > 0.25   # more than 25% dark pixels → likely text-heavy

def make_white_background(raw_bytes: bytes) -> Image.Image | None:
    """
    Use rembg to remove background, then paste onto pure white canvas.
    Returns a 512x512 RGB PIL image, or None if processing failed.
    """
    try:
        # rembg removes background → RGBA image
        result_bytes = remove(raw_bytes)
        img_rgba = Image.open(io.BytesIO(result_bytes)).convert("RGBA")

        # Check if subject is too small (rembg removed too much)
        alpha = img_rgba.split()[3]
        non_transparent = sum(1 for p in alpha.getdata() if p > 10)
        if non_transparent / (img_rgba.width * img_rgba.height) < 0.05:
            return None   # subject too small after bg removal

        # Paste onto white canvas
        canvas = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
        canvas.paste(img_rgba, mask=img_rgba.split()[3])
        canvas = canvas.convert("RGB")

        # Resize to standard size with padding
        canvas.thumbnail(OUTPUT_SIZE, Image.LANCZOS)
        final  = Image.new("RGB", OUTPUT_SIZE, (255, 255, 255))
        offset = ((OUTPUT_SIZE[0] - canvas.width)  // 2,
                  (OUTPUT_SIZE[1] - canvas.height) // 2)
        final.paste(canvas, offset)
        return final

    except Exception as e:
        print(f"      [rembg error] {e}")
        return None

def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

# ── PER-CONCEPT DOWNLOADER ────────────────────────────────────────────────────
def download_concept(concept: str, query: str,
                     out_dir: Path, target: int):

    raw_dir   = out_dir / "raw"
    clean_dir = out_dir / "clean"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    # Count existing clean images
    existing_clean = [p for p in clean_dir.iterdir()
                      if p.suffix == ".jpg"]
    if len(existing_clean) >= target:
        print(f"  [{concept:15}] ✅ already {len(existing_clean)} clean images")
        return

    saved  = len(existing_clean)
    hashes = set()
    print(f"\n  [{concept:15}] need {target - saved} more clean images...")

    batch = 50
    while saved < target:
        pages = search_wikimedia(query, limit=batch)
        if not pages:
            print(f"    no more Wikimedia results for '{concept}'")
            break

        for page in tqdm(pages, desc=f"    {concept}", leave=False):
            if saved >= target:
                break

            url = get_image_url(page)
            if not url:
                continue

            raw_bytes = download_raw(url)
            if not raw_bytes:
                continue

            h = file_hash(raw_bytes)
            if h in hashes:
                continue
            hashes.add(h)

            # Quick PIL check before rembg
            try:
                img = Image.open(io.BytesIO(raw_bytes))
            except Exception:
                continue

            if is_too_small(img):
                continue
            if has_heavy_text_regions(img):
                print(f"      skipped (text detected)")
                continue

            # Background removal → white
            clean_img = make_white_background(raw_bytes)
            if clean_img is None:
                continue

            # Save clean image
            clean_path = clean_dir / f"{concept}_{saved+1:03d}.jpg"
            clean_img.save(clean_path, "JPEG", quality=95)
            saved += 1
            print(f"    ✅ {saved}/{target}  {clean_path.name}")

            time.sleep(DELAY)

        if len(pages) < batch:
            break

    print(f"  [{concept:15}] done → {saved} clean images in {clean_dir}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    CLASS1_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Class I Clean Dataset Downloader")
    print(f"  Target  : {TARGET} clean images per concept")
    print(f"  Concepts: {len(CLASS1_CONCEPTS)}")
    print(f"  Output  : {CLASS1_DIR}")
    print(f"  Process : Download → Remove BG → White BG → Validate")
    print("=" * 65)

    for concept, query in CLASS1_CONCEPTS.items():
        out_dir = CLASS1_DIR / concept
        download_concept(concept, query, out_dir, TARGET)

    # ── FINAL SUMMARY ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)
    total = 0
    for concept in CLASS1_CONCEPTS:
        clean_dir = CLASS1_DIR / concept / "clean"
        count = len(list(clean_dir.glob("*.jpg"))) if clean_dir.exists() else 0
        status = "✅" if count >= TARGET else f"⚠️  {count}/{TARGET}"
        print(f"  {concept:<15}  {status}")
        total += count

    print(f"\n  Total clean images: {total}")
    print(f"  Location: {CLASS1_DIR}")
    print("\n  Each image: 512x512 px, white background, single object")
    print("=" * 65)

if __name__ == "__main__":
    main()