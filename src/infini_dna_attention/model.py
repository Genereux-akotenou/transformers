# path setting
import sys, os, warnings
DEV_FOLDER = "/Users/genereux/Documents/UM6P/COURS-S3/S3-PROJECT/transformers/src"
sys.path.append(os.path.dirname(DEV_FOLDER))
warnings.filterwarnings("ignore")

# package import
from transformers import TrainingArguments
from infini_dna_attention.model import InfiniteEncoderDecoderTransformer
import torch
from torch import nn, optim
from transformers import TrainingArguments
import sklearn
from infini_dna_attention.bert.configuration_bert import BertConfig
from infini_dna_attention.bert.modeling_bert import BertModel, BertForSequenceClassification

config = BertConfig(
    vocab_size=4101,              
    hidden_size=768,               
    num_hidden_layers=12,          
    num_attention_heads=12,        
    intermediate_size=3072,        
    hidden_act="gelu",             
    max_position_embeddings=512,   
    type_vocab_size=2,             
    initializer_range=0.02,        
    layer_norm_eps=1e-12,          
    hidden_dropout_prob=0.1,       
    attention_probs_dropout_prob=0.1, 
    is_decoder=False,              
    batch_size=128,
    segment_size=512,
    position_embedding_type="absolute"
)
#model_infini = BertForSequenceClassification(config)
model_infini = BertModel(config)


# model transfers
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig
bert_model_name = "zhihan1996/DNA_bert_6"
config = BertConfig.from_pretrained(bert_model_name)
model_base  = AutoModel.from_pretrained(bert_model_name, trust_remote_code=True, config=config)
tokenizer = AutoTokenizer.from_pretrained(bert_model_name, trust_remote_code=True)

# embedding
pretrained_state_dict = model_base.embeddings.state_dict()
transformer_state_dict = {}
transformer_state_dict['word_embeddings.weight'] = pretrained_state_dict['word_embeddings.weight']
transformer_state_dict['position_embeddings.weight'] = pretrained_state_dict['position_embeddings.weight']
transformer_state_dict['token_type_embeddings.weight'] = pretrained_state_dict['token_type_embeddings.weight']
transformer_state_dict['LayerNorm.weight'] = pretrained_state_dict['LayerNorm.weight']
transformer_state_dict['LayerNorm.bias'] = pretrained_state_dict['LayerNorm.bias']
model_infini.embeddings.load_state_dict(transformer_state_dict)

# weights
base_state_dict = model_base.state_dict()
infini_state_dict = model_infini.state_dict()
for key in base_state_dict:
    if key in infini_state_dict and infini_state_dict[key].shape == base_state_dict[key].shape:
        infini_state_dict[key] = base_state_dict[key]
    else:
        print(f"Skipping {key} as it does not match or does not exist in the new model.")
model_infini.load_state_dict(infini_state_dict, strict=True)

# test
import random
def generate_dna_kmers(kmer_length: int, num_kmers: int) -> str:
    dna_bases = ['A', 'T', 'G', 'C']
    kmers = [
        ''.join(random.choices(dna_bases, k=kmer_length))
        for _ in range(num_kmers)
    ]
    dna_sequence = ' '.join(kmers)
    return dna_sequence
dna = generate_dna_kmers(kmer_length=6, num_kmers=10000)
inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"]
hidden_states = model_infini(inputs)[0]
embedding_mean = torch.mean(hidden_states[0], dim=0)
print(embedding_mean.shape) # expect to be 768