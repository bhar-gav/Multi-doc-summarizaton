import os
import nltk
import pickle
from nltk.tokenize import word_tokenize
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import torch

# # Ensure NLTK resources are downloaded
# nltk.download('punkt')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    return tokens

def read_documents_from_directory(directory_path):
    documents = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        tokens = preprocess_text(content)
                        documents.append(tokens)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    return documents

class TextDataset(Dataset):
    def __init__(self, documents, vocab):
        self.documents = documents
        self.vocab = vocab

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        tokens = self.documents[idx]
        indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        return torch.tensor(indices, dtype=torch.long)

def collate_fn(batch):
    max_len = max(len(seq) for seq in batch)
    padded_batch = [torch.cat([seq, torch.tensor([vocab['<pad>']] * (max_len - len(seq)))]) for seq in batch]
    return torch.stack(padded_batch)

def save_vocab(vocab, filepath):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary saved to {filepath}")

def load_vocab(filepath):
    with open(filepath, 'rb') as f:
        vocab = pickle.load(f)
    print(f"Vocabulary loaded from {filepath}")
    return vocab

def save_vocab_as_txt(vocab, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for token, index in vocab.items():
            f.write(f"{token}\t{index}\n")
    print(f"Vocabulary saved to {filepath}")

def load_vocab_from_txt(filepath):
    stoi = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            token, index = line.strip().split('\t')
            stoi[token] = int(index)
    print(f"Vocabulary loaded from {filepath}")
    return stoi


directory_path = os.path.join(os.path.dirname(__file__), '..', '..','..', 'datasets', 'DUC-2004-Dataset', 'DUC2004_Summarization_Documents', 'duc2004_testdata', 'tasks1and2')

documents = read_documents_from_directory(directory_path)

# Specify the directory and file paths
vocab_loc = os.path.dirname(__file__)
vocab_pkl_path = os.path.join(vocab_loc, 'vocab.pkl')
vocab_txt_path = os.path.join(vocab_loc, 'vocab.txt')

# Build vocabulary using nltk
all_tokens = [token for doc in documents for token in doc]
freq_dist = nltk.FreqDist(all_tokens)
vocab = {token: index for index, (token, _) in enumerate(freq_dist.items())}

# Add special tokens
vocab['<unk>'] = len(vocab)
vocab['<pad>'] = len(vocab)
print("Vocabulary size:", len(vocab))

# Save the vocabulary
save_vocab(vocab, vocab_pkl_path)
save_vocab_as_txt(vocab, vocab_txt_path)

# Load the vocabulary from .txt (to demonstrate loading)
vocab = load_vocab_from_txt(vocab_txt_path)

# Prepare dataset and dataloader
dataset = TextDataset(documents, vocab)
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

# Iterate over dataloader
for batch in dataloader:
    print("Batch of token indices:")
    print(batch)
