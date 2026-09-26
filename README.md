# Linguistic Biasness — Project Notes

## Assistant instructions

- **Always respond in Hinglish** (Hindi + English mixed, Roman script) in this project, regardless of the language the user types in.

## Problem statement (summary)

Does changing only the language, script, or cultural cue change which culturally
different but semantically valid images a model retrieves? E.g. does "doll" (English)
vs "गुड़िया" (Hindi) vs "gudiya" (Romanized) pull different results from CLIP-family
models, and does adding explicit cultural context ("Indian doll" / "भारतीय गुड़िया" /
"Bhartiya gudiya") change that further?

### Models to test
- English-centric baseline: CLIP, OpenCLIP
- Multilingual: M-CLIP / XLM-R-based CLIP
- Translation/alignment-based: NLLB-CLIP
- Latest VLMs

### Languages
At least 5 Indian languages (+ English as reference): Hindi, Urdu, Bengali, Tamil, Telugu.

### Prompt design
Three script variants per concept (English word, native script word, Romanized word),
each with and without explicit cultural context — to separate *implicit* cultural
association (from language alone) vs *explicit* cultural instruction.

### Dataset
Per concept: Western vs Indian categories, ≥100 images per category.

### Taxonomy
| Class | Description | Examples |
|---|---|---|
| I | Low cultural malleability / semantic controls | apple, tiger, snake, bottle, cloud, ... |
| II | Implicitly culturally malleable | doll, house, kitchen, wedding, clothes, ... |
| III | Strongly culturally embedded | festival, ritual, textile pattern, marriage ceremony, ... |
| IV | Socially sensitive / stereotype-prone | doctor, homemaker, criminal, terrorist, ... |

## Repo state

- `Class1_Data/` — Class I control concepts (30 folders), each with `clean/` containing
  100 white-background validated images. Generated via `download_class1_clean.py`
  (Wikimedia Commons + rembg background removal).
- `download_class1_clean.py` — downloader/cleaner for Class I concept images.
- `run_linguistic_bias.py` — path-blind CLIP/OpenCLIP retrieval pipeline. Currently
  wired for the "Dolls" (Class II) concept only: flattens Indian/European/African/...
  culture subfolders into one anonymized pool, ranks against English vs Hindi prompts,
  then reveals culture labels post-hoc to measure retrieval bias.
- `Problem statement.pdf` — full original problem statement with taxonomy tables and
  positive/negative use-case examples.
