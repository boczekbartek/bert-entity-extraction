import numpy as np

import joblib
import torch
from tqdm import tqdm

import config
import dataset
import engine
from model import EntityModel
from train import process_data
from tqdm import tqdm
from sklearn.metrics import classification_report

if __name__ == "__main__":

    meta_data = joblib.load(config.META_PATH)
    enc_tag = meta_data["enc_tag"]

    num_tag = len(list(enc_tag.classes_))

    device = torch.device("cuda")
    model = EntityModel(num_tag=num_tag)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)
    train_sentences, train_tag, enc_tag = process_data(config.TRAINING_FILE, enc_tag=enc_tag)
    train_dataset = dataset.EntityDataset(texts=train_sentences, tags=train_tag)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, num_workers=1
    )


    test_sentences, test_tag, enc_tag = process_data(config.DEV_FILE, enc_tag=enc_tag)
    valid_dataset = dataset.EntityDataset(texts=test_sentences, tags=test_tag)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, num_workers=1
    )
    true_tags = []
    pred_tags = []
    
    pred_tags_parsed = []
    true_tags_parsed = []
    VERBOSE=False
    with torch.no_grad():
        for i, data in enumerate(valid_data_loader):
            sentence = test_sentences[i]

            # detect the index of the end of the sentence
            tokenized_sentence = data['ids'].numpy().reshape(-1)
            end = [i for i, tok in enumerate(tokenized_sentence) if tok == config.TOKENIZER.vocab['[SEP]']][0]

            # read TRUE labels for this sentence
            true_sent_tags = data['target_tag'].numpy().reshape(-1)[: end][1:]
            true_tags.extend(true_sent_tags)

            token_ids = data['token_ids'].numpy().reshape(-1)[: end][1:]

            true_tag_parsed = []
            prev_id = None
            for label, tok_id in zip(true_sent_tags, token_ids):
                if prev_id == tok_id:
                    prev_id = tok_id
                    continue
                prev_id = tok_id
                true_tag_parsed.append(label)
            true_tags_parsed.extend(true_tag_parsed)
            tokens = config.TOKENIZER.convert_ids_to_tokens(tokenized_sentence, skip_special_tokens=True)
            # print(len(sentence), '|', sentence)
            
            if VERBOSE:
              print(len(tokens), '|', tokens)
              print(len(token_ids), '|', token_ids)
              print(len(true_sent_tags), '|', enc_tag.inverse_transform(true_sent_tags))
              print(len(true_tag_parsed), '|', enc_tag.inverse_transform(true_tag_parsed))
            # Decode the sentence
            
            # Get token ids to glue labels

            for k in ('ids', 'mask', 'token_type_ids', 'target_tag'):
                data[k] = data[k].to(device)
            
            # Query the model for tags predictions
            tag_logits, _ = model(**data)            
            # Get actual tags from logits and strip them to sentence length
            pred_sent_tags = tag_logits.argmax(2).cpu().numpy().reshape(-1)[: end][1:]
            pred_tags.extend(pred_sent_tags)

            # Glue tags that were extended because of tokenization.
            pred_tag_parsed = []
            prev_id = None
            for label, tok_id in zip(pred_sent_tags, token_ids):
                if prev_id == tok_id:
                    prev_id = tok_id
                    continue
                prev_id = tok_id
                pred_tag_parsed.append(label)

            pred_tags_parsed.extend(pred_tag_parsed)
            if VERBOSE:
              print(len(pred_sent_tags), '|', enc_tag.inverse_transform(pred_sent_tags))

              print(len(pred_tag_parsed), '|', enc_tag.inverse_transform(pred_tag_parsed))
              
            if not np.array_equal(true_tag_parsed, pred_tag_parsed):
            # if "n't" in sentence:
            # if '[UNK]' in tokens:
            # unk_i = []
            # for i,t in enumerate(tokens):
            #   if t == '[UNK]':
            #     unk_i.append(i)
              print(f'---------------------{i}--------------------------')
              print(sentence)
              mask=np.array(true_tag_parsed)!=np.array(pred_tag_parsed)
              print(np.ma.array(sentence, mask=~mask))
              # print('UNK:', [sentence[i] for i in unk_i])
              print('TRUE |', enc_tag.inverse_transform(true_tag_parsed))
              print('PRED |', enc_tag.inverse_transform(pred_tag_parsed))


    print(classification_report(true_tags, pred_tags,target_names=enc_tag.classes_))
    print(classification_report(true_tags_parsed, pred_tags_parsed,target_names=enc_tag.classes_))
