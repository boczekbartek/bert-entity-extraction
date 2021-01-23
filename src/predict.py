import numpy as np

import joblib
import torch

import config
import dataset
import engine
from model import EntityModel
from train import process_data
from tqdm import tqdm

if __name__ == "__main__":

    meta_data = joblib.load("meta.bin")
    enc_tag = meta_data["enc_tag"]

    num_tag = len(list(enc_tag.classes_))

    device = torch.device("cuda")
    model = EntityModel(num_tag=num_tag)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)
    test_sentences, test_tag, enc_tag = process_data(config.DEV_FILE, enc_tag=enc_tag)
    valid_dataset = dataset.EntityDataset(texts=test_sentences, tags=test_tag)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, num_workers=1
    )
    results = []
    with torch.no_grad():
        for i, data in enumerate(valid_data_loader):
            for k, v in data.items():
                data[k] = v.to(device)
            sentence = test_sentences[i]
            tokenized_sentence = config.TOKENIZER.encode(sentence)

            tag, _ = model(**data)
            print(f"{i} | {sentence}")
            print(f"{i} | {tokenized_sentence}")
            entities = enc_tag.inverse_transform(
                tag.argmax(2).cpu().numpy().reshape(-1)
            )[: len(tokenized_sentence)][1:-1]
            print(f"{i} | {entities}")
            results.append(entities)
    import pickle as pkl

    with open("results.pkl", "wb") as f:
        pkl.dump(results, f)

