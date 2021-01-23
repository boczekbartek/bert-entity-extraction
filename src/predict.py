import numpy as np

import joblib
import torch

import config
import dataset
import engine
from model import EntityModel
from train import process_data
from tqdm import tqdm
from sklearn.metrics import classification_report

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
    true_tags = []
    pred_tags = []
    with torch.no_grad():
        for i, data in enumerate(valid_data_loader):
            for k, v in data.items():
                data[k] = v.to(device)
            sentence = test_sentences[i]
            tokenized_sentence = config.TOKENIZER.encode(sentence)
            true_sent_tags = data['target_tag'].cpu().numpy().reshape(-1)[: len(tokenized_sentence)][1:-1]
            print(f'---------------------{i}--------------------------')
            print(config.TOKENIZER.convert_ids_to_tokens(tokenized_sentence))
            print(enc_tag.inverse_transform(true_sent_tags))
            tag_logits, _ = model(**data)            
            pred_sent_tags = tag_logits.argmax(2).cpu().numpy().reshape(-1)[: len(tokenized_sentence)][1:-1]
            print(enc_tag.inverse_transform(pred_sent_tags))
            true_tags.extend(true_sent_tags)
            pred_tags.extend(pred_sent_tags)
    
    print(classification_report(true_tags, pred_tags,target_names=enc_tag.classes_))
