import transformers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 8
EPOCHS = 10
# BASE_MODEL_PATH = "distilbert-base-uncased"
BASE_MODEL_PATH = "bert-base-uncased"
MODEL_PATH = "model.v2.bin"
TRAINING_FILE = "../data/SEM-2012-SharedTask-CD-SCO-training-simple-v2.txt"
DEV_FILE = "../data/SEM-2012-SharedTask-CD-SCO-dev-simple-v2.txt"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH, do_lower_case=True
)
