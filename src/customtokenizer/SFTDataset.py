import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, block_size):
        self.data = data
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.assistant_id = tokenizer.convert_tokens_to_ids("<|assistant|>")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]

        text = (
            "<|user|>\n" + ex["prompt"] + "\n"
            "<|assistant|>\n" + ex["response"] + self.tokenizer.eos_token
        )

        ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.block_size + 1
        )

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)

        labels = torch.full_like(y, -100)

        seen_assistant = False
        for i, tok in enumerate(x):
            if tok == self.assistant_id:
                seen_assistant = True
                continue
            if seen_assistant:
                labels[i] = y[i]

        return x, labels
