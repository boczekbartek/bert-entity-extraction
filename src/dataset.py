import config
import torch


class EntityDataset:
    def __init__(self, texts, tags):
        # texts: [["hi", ",", "my", "name", "is", "abhishek"], ["hello".....]]
        # pos/tags: [[1 2 3 4 1 5], [....].....]]
        self.texts = texts
        self.tags = tags

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        tags = self.tags[item]

        ids = []
        target_tag = []
        token_ids = []
        
        for i, s in enumerate(text):
            inputs = config.TOKENIZER.encode(s, add_special_tokens=False)
            input_len = len(inputs)
            ids.extend(inputs)
            target_tag.extend([tags[i]] * input_len)
            token_ids.extend([i]*input_len)

        ids = ids[: config.MAX_LEN - 2]
        target_tag = target_tag[: config.MAX_LEN - 2]
        token_ids = token_ids[: config.MAX_LEN - 2]

        ids = [config.TOKENIZER.vocab['[CLS]']] + ids + [config.TOKENIZER.vocab['[SEP]']]
        target_tag = [0] + target_tag + [0]
        token_ids = [-1] + token_ids + [-1]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = config.MAX_LEN - len(ids)

        ids = ids + ([config.TOKENIZER.vocab['[PAD]']] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)
        token_ids = token_ids + ([-1] * padding_len)

        ret = {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
            "token_ids": torch.tensor(token_ids, dtype=torch.long)
        }
        return ret
