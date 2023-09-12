#preparing the data
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader

raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")

split_datasets = raw_datasets['train'].train_test_split(train_size=0.9, seed=20)
split_datasets["validation"] = split_datasets.pop("test")

save_path = './data/kd4'
split_datasets.save_to_disk(save_path)