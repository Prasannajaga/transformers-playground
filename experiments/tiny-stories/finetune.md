# Dataset Card: `tiny-stories-QA`

> this is dataset created from tiny-stories based QA pairs,  
> have fun doing Supervised Fine Tuning with it

---

### Features

| Feature    | Type     | Description                    |
| :--------- | :------- | :----------------------------- |
| `prompt`   | `string` | Context + user question        |
| `response` | `string` | Target assistant response      |

---

### Statistics

- **Total Training Examples**: `271,779`
- **Task Type**: Supervised Fine-Tuning (SFT)

---

### Quick Start

#### Loading via Hugging Face

```python
from datasets import load_dataset

dataset = load_dataset("prasannaJagadesh/tiny-stories-QA")
```

#### Reading Compressed Data

```python
import gzip
import json

# saved in format like this 
with gzip.open(path, "wt", encoding="utf-8") as f:
    json.dump(data, f)

# you can read like this 
with gzip.open(path, "rt", encoding="utf-8") as f:
    data = json.load(f)
    return data
```

---

### Special Tokens & Template

The dataset follows a specific prompt structure for training:

```python
prompt = (
    "<|story|> \n" +
    story + "\n"
    "<|user|>\n" + question + "\n"
    "<|assistant|>\n"
)
```

---

### Resources

- **Full Training Logs**: [app.log](https://github.com/Prasannajaga/transformers-playground/blob/main/src/models/tiny-stories/app.log)
