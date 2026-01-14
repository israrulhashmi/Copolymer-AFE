# Data

Place your dataset files in the `data/` directory. This repository does not include raw data by default.

Accepted formats:
- A single CSV named `data/dataset.csv` with columns `sequence` and `target` (or `property`).
- Two text files: `data/sequence.txt` and `data/property.txt` (aligned line-by-line).

Example CSV header:
```
sequence,target
AAABBCC, -5.2
AABBC, -4.1
```

Do NOT commit large raw datasets. If you need to store large binary files, consider using Git LFS or GitHub Releases.
