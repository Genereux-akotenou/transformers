# Path Configuration
import sys, os, warnings
DEV_FOLDER = "/Users/genereux/Documents/UM6P/COURS-S3/S3-PROJECT/transformers/src/"
sys.path.append(os.path.dirname(DEV_FOLDER))
warnings.filterwarnings("ignore")

# custom model
from basic_attention.model import EncoderDecoderTransformer
from transfer_learning.trainer import Trainer

# data utils
import torchtext
import torch
from torch import nn
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.vocab import build_vocab_from_iterator
from torchtext.utils import download_from_url, extract_archive
import io
import sklearn
from transformers import TrainingArguments

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
train_urls = ('train.fr.gz', 'train.en.gz')
val_urls   = ('val.fr.gz', 'val.en.gz')
test_urls  = ('test_2016_flickr.fr.gz', 'test_2016_flickr.en.gz')

train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
val_filepaths   = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
test_filepaths  = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

fr_tokenizer = get_tokenizer('spacy', language='fr')
en_tokenizer = get_tokenizer('spacy', language='en')

# def build_vocab(filepath, tokenizer):
#   counter = Counter()
#   with io.open(filepath, encoding="utf8") as f:
#     for string_ in f:
#       counter.update(tokenizer(string_))
#   return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

def build_vocab(filepath, tokenizer):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    
    vocab = build_vocab_from_iterator([counter.keys()], specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab

fr_vocab = build_vocab(train_filepaths[0], fr_tokenizer)
en_vocab = build_vocab(train_filepaths[1], en_tokenizer)

def data_process(filepaths):
  raw_fr_iter = iter(io.open(filepaths[0], encoding="utf8"))
  raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
  data = []
  for (raw_fr, raw_en) in zip(raw_fr_iter, raw_en_iter):
    fr_tensor_ = torch.tensor([fr_vocab[token] for token in fr_tokenizer(raw_fr)],
                            dtype=torch.long)
    en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)],
                            dtype=torch.long)
    #data.append((fr_tensor_, en_tensor_))
    data.append({"input": raw_fr, "target": raw_en,"input_ids": fr_tensor_, "labels": en_tensor_})
  return data

train_data = data_process(train_filepaths)
val_data   = data_process(val_filepaths)
test_data  = data_process(test_filepaths)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_IDX = fr_vocab['<pad>']
BOS_IDX = fr_vocab['<bos>']
EOS_IDX = fr_vocab['<eos>']

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def generate_batch(data_batch):
    input_ids_batch, labels_batch = [], []
    for data in data_batch:
        input_ids = data["input_ids"]
        labels = data["labels"]
        input_ids_batch.append(torch.cat([torch.tensor([BOS_IDX]), input_ids, torch.tensor([EOS_IDX])], dim=0))
        labels_batch.append(torch.cat([torch.tensor([BOS_IDX]), labels, torch.tensor([EOS_IDX])], dim=0))

    input_ids_batch = pad_sequence(input_ids_batch, batch_first=True, padding_value=PAD_IDX)
    labels_batch = pad_sequence(labels_batch, batch_first=True, padding_value=PAD_IDX)
    return {
        "input_ids": input_ids_batch,
        "labels": labels_batch,
    }

# def generate_batch(data_batch):
#   data_batch
#   inputs, labels = [], []
#   for (inputs, labels) in data_batch:
#     inputs_batch.append(torch.cat([torch.tensor([BOS_IDX]), inputs, torch.tensor([EOS_IDX])], dim=0))
#     labels_batch.append(torch.cat([torch.tensor([BOS_IDX]), labels, torch.tensor([EOS_IDX])], dim=0))
#   inputs_batch = pad_sequence(inputs_batch, padding_value=PAD_IDX)
#   labels_batch = pad_sequence(labels_batch, padding_value=PAD_IDX)
#   return inputs_batch, labels_batch

# train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
#                         shuffle=True, collate_fn=generate_batch)
# valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
#                         shuffle=True, collate_fn=generate_batch)
# test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
#                        shuffle=True, collate_fn=generate_batch)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# optimizer parameter setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epoch = 1000
clip = 1.0
weight_frcay = 5e-4
inf = float('inf')

# vocab setting
src_pad_idx = PAD_IDX
trg_pad_idx = PAD_IDX
trg_sos_idx = BOS_IDX
enc_voc_size = len(fr_vocab)
dec_voc_size = len(en_vocab)

# Encoder decoder
model = EncoderDecoderTransformer(
    src_pad_idx=src_pad_idx,
    trg_pad_idx=trg_pad_idx,
    trg_sos_idx=trg_sos_idx,
    d_model=512,
    enc_voc_size=enc_voc_size,
    dec_voc_size=dec_voc_size,
    max_len=256,
    ffn_hidden=2048,
    n_head=8,
    n_layers=6,
    drop_prob=0.1,
    device=device
).to(device)

# init model weights
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)
model.apply(initialize_weights)

# model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
class CustomTrainingArguments(TrainingArguments):
    def __init__(self, *args, adam_eps=None, factor=None, patience=None, src_pad_idx=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.adam_eps = adam_eps
        self.factor = factor
        self.patience = patience
        self.src_pad_idx = src_pad_idx

def compute_metrics(preds, labels):
    """
    Compute BLEU score and accuracy for predictions and labels.
    Args:
        preds: List of predicted sequences (token IDs).
        labels: List of reference sequences (token IDs).
    Returns:
        Dict with BLEU score and accuracy.
    """
    from sklearn.metrics import accuracy_score
    import evaluate

    bleu_metric = evaluate.load("bleu")

    # Ensure preds and labels are lists of sequences
    if not isinstance(preds[0], list):
        preds = [preds]
    if not isinstance(labels[0], list):
        labels = [labels]

    # Prepare predictions and references for BLEU
    predictions = [" ".join(map(str, pred)) for pred in preds]
    references = [[" ".join(map(str, label))] for label in labels]

    # Compute BLEU score
    bleu_result = bleu_metric.compute(predictions=predictions, references=references)

    # Flatten preds and labels for accuracy
    #flattened_preds = [idx for seq in preds for idx in seq if idx not in {PAD_IDX, EOS_IDX, BOS_IDX}]
    #flattened_labels = [idx for seq in labels for idx in seq if idx not in {PAD_IDX, EOS_IDX, BOS_IDX}]

    # Compute accuracy
    #accuracy = accuracy_score(flattened_labels, flattened_preds)

    return {
        "bleu_score": bleu_result["bleu"],
    #    "accuracy": accuracy,
    }

training_args = CustomTrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    learning_rate=5e-5,
    weight_decay=0.01,
    adam_eps=1e-8,
    factor=0.1,
    patience=2,
    src_pad_idx=src_pad_idx,
    max_grad_norm=1.0,
    #eval_steps=5,
)

trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    training_args=training_args,
    compute_metrics=compute_metrics,
    collate_fn=generate_batch
)
trainer.train()