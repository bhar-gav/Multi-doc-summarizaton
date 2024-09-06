import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize

import pickle

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout=0.5, pretrained_embeddings=None, freeze_embeddings=False):
        super(BiLSTMModel, self).__init__()
        
        # Word Encoding Layer (Embedding Layer)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # If pre-trained embeddings are provided, load them
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False  # Optionally freeze the embeddings

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            bidirectional=True,
                            dropout=dropout,
                            batch_first=True)
        
        # Ranking Layer (Fully Connected Layer)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Step 1: Word Encoding Layer
        x = self.embedding(x)
        
        # Step 2: LSTM Layer
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Step 3: Apply dropout to the output of the LSTM
        lstm_out = self.dropout(lstm_out)
        
        # Step 4: Aggregate LSTM output for the ranking layer
        lstm_out = lstm_out[:, -1, :]  # [batch_size, hidden_dim * 2]
        
        # Step 5: Output Layer
        out = self.fc(lstm_out)
        
        return out

class TextSummarizationDataset(Dataset):
    def __init__(self, data_path, vocab, max_seq_length):
        self.data_path = data_path
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.data = self.load_data()
        
        # Check if the dataset is empty
        if len(self.data) == 0:
            raise ValueError("Dataset is empty. Please check the data files.")
        
def load_data(self):
    texts = []
    labels = []
    for root, _, files in os.walk(self.data_path):
        print(f"Checking directory: {root}, Files: {files}")
        for file_name in files:
            file_path = os.path.join(root, file_name)
            print(f"Reading file: {file_path}")
            try:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        parts = line.strip().split('\t')
                        print(f"Line parts: {parts}")  # Debugging line
                        if len(parts) >= 2:
                            text, label = parts[0], parts[1]
                            texts.append(text)
                            labels.append(label)
                        else:
                            print(f"Skipping malformed line: {line}")
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    
    print(f"Loaded {len(texts)} texts and {len(labels)} labels")
    return list(zip(texts, labels))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        tokens = word_tokenize(text.lower())
        indexed_tokens = [self.vocab.get(token, 0) for token in tokens[:self.max_seq_length]]
        indexed_tokens += [0] * (self.max_seq_length - len(indexed_tokens))  # padding
        label = int(label)  # Assuming binary classification
        return torch.tensor(indexed_tokens), torch.tensor(label, dtype=torch.float)


def train_model(model, dataloader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")


def main():
    # Hyperparameters
    vocab_size = 10000  # Example vocab size
    embedding_dim = 300
    hidden_dim = 128
    num_layers = 2
    num_classes = 1  # Binary classification
    dropout = 0.5
    max_seq_length = 50
    batch_size = 32
    num_epochs = 5

    
    vocab_file_path = os.path.join(os.path.dirname(__file__), 'vocab','vocab.pkl')


    try:
        with open(vocab_file_path, 'rb') as f:
            vocab = pickle.load(f)
        print("File loaded successfully.")
    except FileNotFoundError:
        print("The file was not found.")
    except pickle.UnpicklingError:
        print("Error unpickling the file. The file may be corrupted or not a valid pickle file.")
    except EOFError:
        print("Reached end of file unexpectedly. The file might be incomplete.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


    # Print some information about the loaded vocabulary
    print("Vocabulary loaded.")
    print("Vocabulary size:", len(vocab))
    # print("Sample of vocabulary:", list(vocab.items())[:1])  # Print a sample of the vocabulary

    # Prepare dataset and dataloader
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'preprocessed_data')
    print("Data Path:", data_path)
    dataset = TextSummarizationDataset(data_path, vocab, max_seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = BiLSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout)
    criterion = nn.BCEWithLogitsLoss()  # Binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, num_epochs)


if __name__ == "__main__":
    main()
