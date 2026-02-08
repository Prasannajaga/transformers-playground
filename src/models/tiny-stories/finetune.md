# Dataset Card for tiny-stories-QA

this is dataset created from tiny-stories based QA pairs, 
have fun doing Supervised Fine Tuning with it

## Features

- `prompt`: string
- `response`: string

## Statistics

- **Training examples**: 271,779

## Quick Start

```python
from datasets import load_dataset

dataset = load_dataset("prasannaJagadesh/tiny-stories-QA")

# saved in format like this 
with gzip.open(path, "wt", encoding="utf-8") as f:
    json.dump(data, f)

# you can read like this 
with gzip.open(path, "rt", encoding="utf-8") as f:
    data = json.load(f)
return data
```

