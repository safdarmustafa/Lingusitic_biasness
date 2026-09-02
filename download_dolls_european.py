import csv
import hashlib
import io
import time
from pathlib import Path

import requests
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path.home() / "Desktop" / "Cultural_CLIP_Pilot"
OUTPUT_DIR = BASE_DIR / "Dolls" / "European"

TARGET_IMAGES = 30

COMMONS_API = "https://commons.wikimedia.org/w/api.php"

HEADERS = {
    "User-Agent": (
        "CulturalCLIPPilot/1.0 "
        "Academic research image collection"
    )
}


# ============================================================
# VERY SPECIFIC EUROPEAN/WESTERN DOLL SEARCHES
# ============================================================

SEARCH_QUERIES = [
    # General European folk dolls
    "European folk doll",
    "European traditional doll",
    "European folk dolls",
    "European traditional dolls",

    # German
    "German traditional doll",
    "German folk doll",
    "German folk dolls",
    "German costume doll",
    "German traditional costume doll",

    # French
    "French traditional doll",
    "French folk doll",
    "French folk dolls",
    "French costume doll",
    "French porcelain doll",

    # British / English
    "British traditional doll",
    "British folk doll",
    "English traditional doll",
    "English folk doll",

    # Dutch
    "Dutch traditional doll",
    "Dutch folk doll",
    "Dutch costume doll",

    # Scandinavian
    "Swedish traditional doll",
    "Swedish folk doll",
    "Norwegian traditional doll",
    "Norwegian folk doll",
    "Danish traditional doll",
    "Danish folk doll",

    # Eastern/Central European
    "Polish traditional doll",
    "Polish folk doll",
    "Russian traditional doll",
    "Russian folk doll",
    "Ukrainian traditional doll",
    "Ukrainian folk doll",

    # Material/style
    "European porcelain doll",
    "European wooden doll",
    "European costume doll",
    "European folk costume doll",
]


# ============================================================
# WORD FILTERS
# ============================================================

# At least one of these MUST appear in the title/description.
DOLL_WORDS = [
    "doll",
    "dolls",
    "puppet",
    "puppets",
    "figurine doll",
    "folk doll",
    "costume doll",
    "porcelain doll",
    "wooden doll",
]


# Reject obvious irrelevant results.
EXCLUDE_WORDS = [
    "cat",
    "dog",
    "horse",
    "bird",
    "animal",
    "person",
    "people",
    "man",
    "woman",
    "boy",
    "girl",
    "portrait",
    "selfie",
    "statue",
    "sculpture",
    "monument",
    "painting",
    "drawing",
    "illustration",
    "dollhouse",
    "doll house",
    "dollhouse furniture",
    "toy house",
]


# ============================================================
# HELPERS
# ============================================================

def get_text(candidate):
    """
    Combine Wikimedia title and description into searchable text.
    """

    return (
        candidate.get("title", "") + " " +
        candidate.get("description", "")
    ).lower()


def is_valid_candidate(candidate):
    """
    Strict textual filtering.
    """

    text = get_text(candidate)

    # Must explicitly mention a doll-related term.
    has_doll_word = any(
        word in text
        for word in DOLL_WORDS
    )

    if not has_doll_word:
        return False

    # Reject obvious irrelevant subjects.
    for word in EXCLUDE_WORDS:

        if word in text:
            return False

    return True


def image_hash(image_bytes):
    """
    Generate a hash so duplicate images aren't downloaded.
    """

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        image.thumbnail((256, 256))

        return hashlib.md5(
            image.tobytes()
        ).hexdigest()

    except Exception:

        return hashlib.md5(
            image_bytes
        ).hexdigest()


def search_commons(query, limit=50):

    params = {
        "action": "query",
        "format": "json",
        "generator": "search",

        # Search only Wikimedia Commons files.
        "gsrnamespace": 6,

        "gsrsearch": query,
        "gsrlimit": min(limit, 50),

        "prop": "imageinfo",

        "iiprop": (
            "url|size|mime|extmetadata"
        ),

        # Request a reasonably large preview.
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

    pages = (
        data
        .get("query", {})
        .get("pages", {})
    )

    results = []

    for page in pages.values():

        imageinfo = page.get(
            "imageinfo",
            []
        )

        if not imageinfo:
            continue

        info = imageinfo[0]

        mime = info.get(
            "mime",
            ""
        )

        if not mime.startswith(
            "image/"
        ):
            continue

        metadata = info.get(
            "extmetadata",
            {}
        )

        title = page.get(
            "title",
            ""
        )

        description = metadata.get(
            "ImageDescription",
            {}
        ).get(
            "value",
            ""
        )

        source_url = (
            "https://commons.wikimedia.org/wiki/"
            + title.replace(" ", "_")
        )

        candidate = {
            "title": title,
            "image_url": (
                info.get("thumburl")
                or info.get("url")
            ),
            "source_url": source_url,
            "author": metadata.get(
                "Artist",
                {}
            ).get(
                "value",
                ""
            ),
            "license": metadata.get(
                "LicenseShortName",
                {}
            ).get(
                "value",
                ""
            ),
            "license_url": metadata.get(
                "LicenseUrl",
                {}
            ).get(
                "value",
                ""
            ),
            "description": description,
        }

        # STRICT FILTER
        if is_valid_candidate(candidate):
            results.append(candidate)

    return results


def download_image(url):

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "Content-Type",
        ""
    )

    if not content_type.startswith(
        "image/"
    ):
        return None

    return response.content


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_file = (
        BASE_DIR
        / "Dolls"
        / "european_dolls_metadata.csv"
    )

    # --------------------------------------------------------
    # Existing images
    # --------------------------------------------------------

    existing_files = list(
        OUTPUT_DIR.glob("*.jpg")
    )

    downloaded_count = len(
        existing_files
    )

    print()
    print("=" * 70)
    print("EUROPEAN / WESTERN DOLL COLLECTION")
    print("=" * 70)
    print(
        f"Existing images: "
        f"{downloaded_count}"
    )
    print(
        f"Target images: "
        f"{TARGET_IMAGES}"
    )
    print("=" * 70)

    if downloaded_count >= TARGET_IMAGES:

        print(
            "Already have 30 images."
        )

        return

    # --------------------------------------------------------
    # Existing hashes
    # --------------------------------------------------------

    existing_hashes = set()

    if metadata_file.exists():

        with open(
            metadata_file,
            "r",
            encoding="utf-8",
            newline=""
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                if row.get(
                    "image_hash"
                ):
                    existing_hashes.add(
                        row["image_hash"]
                    )

    # --------------------------------------------------------
    # Collect candidates
    # --------------------------------------------------------

    candidates = {}

    for query in SEARCH_QUERIES:

        print()
        print(
            f"Searching: {query}"
        )

        try:

            results = search_commons(
                query,
                limit=50
            )

            print(
                f"Valid candidates: "
                f"{len(results)}"
            )

            for result in results:

                key = result[
                    "source_url"
                ]

                if key not in candidates:

                    candidates[key] = result

        except Exception as e:

            print(
                f"Search error: {e}"
            )

        time.sleep(0.7)

    candidates = list(
        candidates.values()
    )

    print()
    print(
        "=" * 70
    )
    print(
        f"TOTAL UNIQUE CANDIDATES: "
        f"{len(candidates)}"
    )
    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Metadata CSV
    # --------------------------------------------------------

    file_exists = metadata_file.exists()

    with open(
        metadata_file,
        "a",
        encoding="utf-8",
        newline=""
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
            fieldnames=fieldnames
        )

        if not file_exists:
            writer.writeheader()

        # ----------------------------------------------------
        # Download
        # ----------------------------------------------------

        for candidate in candidates:

            if downloaded_count >= TARGET_IMAGES:
                break

            try:

                image_bytes = download_image(
                    candidate[
                        "image_url"
                    ]
                )

                if not image_bytes:
                    continue

                h = image_hash(
                    image_bytes
                )

                # Duplicate check.
                if h in existing_hashes:
                    continue

                image = Image.open(
                    io.BytesIO(
                        image_bytes
                    )
                )

                width, height = image.size

                # Reject tiny images.
                if width < 400 or height < 400:
                    continue

                image = image.convert(
                    "RGB"
                )

                filename = (
                    f"european_doll_"
                    f"{downloaded_count + 1:03d}.jpg"
                )

                filepath = (
                    OUTPUT_DIR
                    / filename
                )

                image.save(
                    filepath,
                    "JPEG",
                    quality=95
                )

                writer.writerow({
                    "filename": filename,
                    "culture": "European/Western",
                    "search_query": "",
                    "title": candidate[
                        "title"
                    ],
                    "source_url": candidate[
                        "source_url"
                    ],
                    "author": candidate[
                        "author"
                    ],
                    "license": candidate[
                        "license"
                    ],
                    "license_url": candidate[
                        "license_url"
                    ],
                    "image_hash": h,
                })

                csv_file.flush()

                existing_hashes.add(h)

                downloaded_count += 1

                print(
                    f"[{downloaded_count:02d}/"
                    f"{TARGET_IMAGES}] "
                    f"{filename}"
                )

                print(
                    f"    {candidate['title']}"
                )

                time.sleep(0.5)

            except Exception as e:

                print(
                    f"Skipped candidate: {e}"
                )

    print()
    print("=" * 70)
    print(
        "EUROPEAN DOLL DOWNLOAD FINISHED"
    )
    print("=" * 70)
    print(
        f"Downloaded: "
        f"{downloaded_count}/"
        f"{TARGET_IMAGES}"
    )

    print(
        f"Folder: "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Metadata: "
        f"{metadata_file}"
    )


if __name__ == "__main__":
    main()
