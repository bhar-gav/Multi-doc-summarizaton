import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import pickle

# Define the BiLSTM model
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.5, pretrained_embeddings=None, freeze_embeddings=False):
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
        self.fc = nn.Linear(hidden_dim * 2, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Step 1: Word Encoding Layer
        x = self.embedding(x)
        
        # Step 2: LSTM Layer
        lstm_out, _ = self.lstm(x)
        
        # Step 3: Apply dropout to the output of the LSTM
        lstm_out = self.dropout(lstm_out)
        
        # Step 4: Aggregate LSTM output for the ranking layer
        lstm_out = lstm_out[:, -1, :]  # [batch_size, hidden_dim * 2]
        
        # Step 5: Output Layer
        out = self.fc(lstm_out)
        
        return out

# Define the Dataset class
class TextSummarizationDataset(Dataset):
    def __init__(self, data_path, vocab, max_seq_length):
        self.data_path = data_path
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.data = self.load_data()
        
        # Check if the dataset is empty
        if len(self.data) == 0:
            raise ValueError("Dataset is empty. Please check the data files.")
        
        # Ensure special tokens exist in the vocabulary
        if '<unk>' not in self.vocab:
            raise ValueError("Vocabulary must contain the '<unk>' token.")
        if '<pad>' not in self.vocab:
            raise ValueError("Vocabulary must contain the '<pad>' token.")
        
    def load_data(self):
        texts = []
        for root, _, files in os.walk(self.data_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        content = file.read()
                        tokens = preprocess_text(content)
                        texts.append(tokens)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
        
        return texts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        indexed_tokens = [self.vocab.get(token, self.vocab.get('<unk>', 0)) for token in tokens[:self.max_seq_length]]
        indexed_tokens += [self.vocab.get('<pad>', 0)] * (self.max_seq_length - len(indexed_tokens))
        return torch.tensor(indexed_tokens, dtype=torch.long)

# Define the training function
def train_model(model, dataloader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs in dataloader:
            labels = torch.zeros(inputs.size(0), 1)  # Dummy labels (for demonstration purposes)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if outputs.size() != labels.size():
                outputs = outputs.view_as(labels)  # Reshape outputs to match labels' size
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")

# Define the preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    return tokens

# Main function
def main():
    # Hyperparameters
    vocab_size = 16155  # Example vocab size from the loaded vocabulary
    embedding_dim = 300
    hidden_dim = 128
    num_layers = 2
    dropout = 0.5
    max_seq_length = 50
    batch_size = 32
    num_epochs = 5

    # Load vocabulary
    vocab_file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'preprocessed_data', 'vocab.pkl')
    try:
        with open(vocab_file_path, 'rb') as f:
            vocab = pickle.load(f)
        print("Vocabulary loaded.")
    except FileNotFoundError:
        print("The file was not found.")
        return
    except pickle.UnpicklingError:
        print("Error unpickling the file. The file may be corrupted or not a valid pickle file.")
        return
    except EOFError:
        print("Reached end of file unexpectedly. The file might be incomplete.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    # Prepare dataset and dataloader
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'preprocessed_data')
    dataset = TextSummarizationDataset(data_path, vocab, max_seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = BiLSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
    criterion = nn.MSELoss()  # Using MSELoss for demonstration; adjust as needed
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, num_epochs)

if __name__ == "__main__":
    main()