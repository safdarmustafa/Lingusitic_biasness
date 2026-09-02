import csv
import hashlib
import io
import os
import time
from pathlib import Path

import requests
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path.home() / "Desktop" / "Cultural_CLIP_Pilot"
DOLLS_DIR = BASE_DIR / "Dolls"

TARGET_PER_GROUP = 30

COMMONS_API = "https://commons.wikimedia.org/w/api.php"

HEADERS = {
    "User-Agent": (
        "CulturalCLIPPilot/1.0 "
        "(academic research image collection; "
        "contact information not provided)"
    )
}

CULTURES = {
    "Indian": [
        "Indian traditional doll",
        "Indian folk doll",
        "Indian traditional dolls",
        "Rajasthani doll",
        "Kathputli doll",
    ],
    "Japanese": [
        "Japanese traditional doll",
        "Japanese folk doll",
        "Japanese traditional dolls",
        "Kokeshi doll",
        "Hina doll",
    ],
    "African": [
        "African traditional doll",
        "African folk doll",
        "African traditional dolls",
        "African doll",
        "African cultural doll",
    ],
    "European": [
        "European traditional doll",
        "European folk doll",
        "European traditional dolls",
        "European folk dolls",
        "traditional European doll",
    ],
}


# ============================================================
# HELPERS
# ============================================================

def image_hash(image_bytes):
    """Return a perceptual-ish MD5 hash of normalized image pixels."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.thumbnail((256, 256))
        return hashlib.md5(image.tobytes()).hexdigest()
    except Exception:
        return hashlib.md5(image_bytes).hexdigest()


def search_commons(query, limit=50):
    """
    Search Wikimedia Commons for image files.
    Returns metadata for candidate images.
    """

    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": min(limit, 50),
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 1200,
    }

    response = requests.get(
        COMMONS_API,
        params=params,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    pages = data.get("query", {}).get("pages", {})

    results = []

    for page in pages.values():
        imageinfo = page.get("imageinfo", [])

        if not imageinfo:
            continue

        info = imageinfo[0]

        mime = info.get("mime", "")

        if not mime.startswith("image/"):
            continue

        extmetadata = info.get("extmetadata", {})

        title = page.get("title", "")

        description_url = (
            "https://commons.wikimedia.org/wiki/"
            + title.replace(" ", "_")
        )

        results.append({
            "title": title,
            "image_url": info.get("thumburl") or info.get("url"),
            "source_url": description_url,
            "author": extmetadata.get("Artist", {}).get("value", ""),
            "license": extmetadata.get("LicenseShortName", {}).get(
                "value", ""
            ),
            "license_url": extmetadata.get("LicenseUrl", {}).get(
                "value", ""
            ),
            "description": extmetadata.get("ImageDescription", {}).get(
                "value", ""
            ),
        })

    return results


def download_image(url):
    """Download an image and return its bytes."""

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")

    if not content_type.startswith("image/"):
        return None

    return response.content


# ============================================================
# MAIN
# ============================================================

def main():

    DOLLS_DIR.mkdir(parents=True, exist_ok=True)

    metadata_file = DOLLS_DIR / "dolls_metadata.csv"

    existing_hashes = set()

    # Read existing metadata if the script is being resumed.
    if metadata_file.exists():
        with open(
            metadata_file,
            "r",
            encoding="utf-8",
            newline="",
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:
                if row.get("image_hash"):
                    existing_hashes.add(row["image_hash"])

    file_exists = metadata_file.exists()

    with open(
        metadata_file,
        "a",
        encoding="utf-8",
        newline="",
    ) as csv_file:

        fieldnames = [
            "filename",
            "culture",
            "search_query",
            "title",
            "source_url",
            "author",
            "license",
            "license_url",
            "image_hash",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        for culture, queries in CULTURES.items():

            output_dir = DOLLS_DIR / culture
            output_dir.mkdir(parents=True, exist_ok=True)

            existing_files = list(
                output_dir.glob("*.jpg")
            )

            downloaded_count = len(existing_files)

            print()
            print("=" * 60)
            print(f"{culture} DOLLS")
            print(f"Already downloaded: {downloaded_count}")
            print(f"Target: {TARGET_PER_GROUP}")
            print("=" * 60)

            if downloaded_count >= TARGET_PER_GROUP:
                print("Target already reached. Skipping.")
                continue

            candidates = []

            for query in queries:

                print(f"Searching Commons: {query}")

                try:
                    results = search_commons(
                        query,
                        limit=50,
                    )

                    candidates.extend(results)

                except Exception as e:
                    print(f"Search error: {e}")

                time.sleep(1)

            # Remove duplicate Commons pages.
            unique_candidates = {}

            for candidate in candidates:
                key = candidate["source_url"]

                if key not in unique_candidates:
                    unique_candidates[key] = candidate

            candidates = list(unique_candidates.values())

            print(
                f"Found {len(candidates)} unique candidates."
            )

            for candidate in candidates:

                if downloaded_count >= TARGET_PER_GROUP:
                    break

                try:

                    image_bytes = download_image(
                        candidate["image_url"]
                    )

                    if not image_bytes:
                        continue

                    # Check duplicate image content.
                    h = image_hash(image_bytes)

                    if h in existing_hashes:
                        continue

                    image = Image.open(
                        io.BytesIO(image_bytes)
                    )

                    # Reject extremely small images.
                    width, height = image.size

                    if width < 300 or height < 300:
                        continue

                    image = image.convert("RGB")

                    filename = (
                        f"{culture.lower()}_"
                        f"{downloaded_count + 1:03d}.jpg"
                    )

                    filepath = output_dir / filename

                    image.save(
                        filepath,
                        "JPEG",
                        quality=95,
                    )

                    writer.writerow({
                        "filename": filename,
                        "culture": culture,
                        "search_query": "",
                        "title": candidate["title"],
                        "source_url": candidate["source_url"],
                        "author": candidate["author"],
                        "license": candidate["license"],
                        "license_url": candidate["license_url"],
                        "image_hash": h,
                    })

                    csv_file.flush()

                    existing_hashes.add(h)

                    downloaded_count += 1

                    print(
                        f"[{downloaded_count:02d}/{TARGET_PER_GROUP}] "
                        f"{filename}"
                    )

                    time.sleep(0.5)

                except Exception as e:
                    print(
                        f"Skipped image because of error: {e}"
                    )

            print(
                f"{culture}: downloaded "
                f"{downloaded_count}/{TARGET_PER_GROUP}"
            )

    print()
    print("=" * 60)
    print("DOLL COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Dataset location: {DOLLS_DIR}")
    print(f"Metadata: {metadata_file}")


if __name__ == "__main__":
    main()
