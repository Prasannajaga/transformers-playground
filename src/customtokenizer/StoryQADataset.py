from torch.utils.data import IterableDataset 
import torch 

class StoryQADataset(IterableDataset):
    def __init__(self, data, tokenizer, block_size):
        self.data = data
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.assistant_id = tokenizer.convert_tokens_to_ids("<|assistant|>")

    def __iter__(self):
        for item in self.data:
            story = item["story"]

            for qa in item["qa"]:
                text = (
                    "<|story|>\n" + story + "\n"
                    "<|user|>\n" + qa["q"] + "\n"
                    "<|assistant|>\n" + qa["a"] + self.tokenizer.eos_token
                )

                ids = self.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.block_size + 1
                )

                x = torch.tensor(ids[:-1])
                y = torch.tensor(ids[1:])

                labels = torch.full_like(y, -100)

                seen_assistant = False
                for i, tok in enumerate(x):
                    if tok == self.assistant_id:
                        seen_assistant = True
                        continue
                    if seen_assistant:
                        labels[i] = y[i]

                yield x, labels


def collate_storyqa(batch):
    """
    batch: list of (x, y) tuples with variable lengths
    """
    xs, ys = zip(*batch)

    max_len = max(x.size(0) for x in xs)

    padded_x = torch.zeros(len(xs), max_len, dtype=torch.long)
    padded_y = torch.full((len(ys), max_len), -100, dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        padded_x[i, : x.size(0)] = x
        padded_y[i, : y.size(0)] = y

    return padded_x, padded_y