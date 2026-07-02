# Dataset

**Source:** Kaggle — "Chest CT-Scan images Dataset" (4-class lung cancer CT scans)

Classes:
- adenocarcinoma
- large.cell.carcinoma
- squamous.cell.carcinoma
- normal

## Setup

1. Download the dataset from Kaggle.
2. Unzip it so you end up with this structure:

```
data/raw/
  adenocarcinoma/
  large.cell.carcinoma/
  squamous.cell.carcinoma/
  normal/
```

3. Folder names must match exactly what's used in `ml/explore_data.py` and
   `ml/preprocess.py` (`CLASSES` list) — rename Kaggle's folders if they differ.

4. Run the pipeline in this order:
   ```
   python ml/explore_data.py     # sanity check + class counts
   python ml/preprocess.py       # -> data/X_all.npy, data/y_all.npy
   python ml/partition.py        # -> data/node1/, node2/, node3/, test/
   python ml/train_centralized.py  # -> models/centralized_model.h5
   ```
