# Data

| Study | File | Where it comes from | In the repo? |
|---|---|---|---|
| B - Framingham | `data/raw/framingham_heart_disease.csv` | Framingham Heart Study (public CSV) | yes |
| A - Kaggle Cardiovascular Disease | `data/raw/cardio_train.csv` | Kaggle, "Cardiovascular Disease dataset" by Svetlana Ulianova | **no** (git-ignored; download it) |

## Getting the Kaggle file

1. Open Kaggle, search for "Cardiovascular Disease dataset" (author Svetlana Ulianova) and download the zip (free Kaggle login needed).
2. Unzip it and save the CSV as `data/raw/cardio_train.csv` (semicolon-separated; columns
   `id;age;gender;height;weight;ap_hi;ap_lo;cholesterol;gluc;smoke;alco;active;cardio`).
3. Any other location works too: pass `--data <path>` to the scripts.

`data/processed/` holds the train/test CSVs written by `src/prepare_data.py` (the Framingham baseline scripts).
