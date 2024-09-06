import os
import re
import pickle
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the stemmer and stop words list
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_document(document):
    """
    Preprocess a single document: lowercase, remove special characters, tokenize, remove stopwords, and stem.
    """
    logging.debug("Starting document preprocessing.")
    
    document = document.lower()
    logging.debug(f"Lowercased document: {document[:100]}...")  # Log the first 100 chars

    document = re.sub(r'[^a-z\s]', '', document)
    logging.debug(f"Cleaned document: {document[:100]}...")  # Log the first 100 chars

    words = word_tokenize(document)
    logging.debug(f"Tokenized words: {words[:10]}...")  # Log the first 10 tokens

    processed_words = [stemmer.stem(word) for word in words if word not in stop_words]
    logging.debug(f"Processed words: {processed_words[:10]}...")  # Log the first 10 tokens

    return processed_words

def preprocess_dataset(dataset_folder, output_folder):
    """
    Process all documents in dataset_folder and save preprocessed documents to output_folder.
    """
    preprocessed_data = {}
    vocabulary = set()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logging.debug(f"Created output folder: {output_folder}")

    logging.info(f"Starting preprocessing of files in {dataset_folder}")

    for root, dirs, files in os.walk(dataset_folder):
        logging.debug(f"Walking through directory: {root}")
        for file in files:
            if file.endswith(".txt"):  # Process only .txt files
                file_path = os.path.join(root, file)
                logging.debug(f"Found file: {file_path}")
                try:
                    logging.info(f"Processing file: {file_path}")
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        document = f.read()
                    
                    logging.debug(f"Read document: {document[:100]}...")  # Log the first 100 chars

                    preprocessed_doc = preprocess_document(document)

                    # Update vocabulary
                    vocabulary.update(preprocessed_doc)
                    logging.debug(f"Updated vocabulary with {len(preprocessed_doc)} words.")

                    # Construct the output file path, preserving the directory structure
                    relative_path = os.path.relpath(file_path, dataset_folder)
                    output_file_path = os.path.join(output_folder, relative_path)
                    logging.debug(f"Output file path: {output_file_path}")

                    # Ensure output directory exists
                    output_file_dir = os.path.dirname(output_file_path)
                    if not os.path.exists(output_file_dir):
                        os.makedirs(output_file_dir)
                        logging.debug(f"Created output directory: {output_file_dir}")

                    # Save the preprocessed document
                    with open(output_file_path, 'w', encoding='utf-8') as out_file:
                        out_file.write(' '.join(preprocessed_doc))

                    preprocessed_data[file_path] = preprocessed_doc
                    logging.info(f"Processed and saved: {output_file_path}")

                except Exception as e:
                    logging.error(f"Error processing file {file_path}: {e}")

    logging.info(f"Preprocessing completed. Processed files are saved in {output_folder}.")

    # Save vocabulary to files
    save_vocabulary(vocabulary, output_folder)
    
    return preprocessed_data

def save_vocabulary(vocabulary, output_folder):
    """
    Save the vocabulary to vocab.pkl and vocab.txt files.
    """
    vocab_pkl_path = os.path.join(output_folder, 'vocab.pkl')
    vocab_txt_path = os.path.join(output_folder, 'vocab.txt')

    # Create a dictionary with token indices
    vocab_dict = {word: idx for idx, word in enumerate(sorted(vocabulary))}
    
    # Add <unk> and <pad> tokens to the vocabulary
    if '<unk>' not in vocab_dict:
        vocab_dict['<unk>'] = len(vocab_dict)
    if '<pad>' not in vocab_dict:
        vocab_dict['<pad>'] = len(vocab_dict)

    logging.debug(f"Saving vocabulary to pickle file: {vocab_pkl_path}")
    # Save vocabulary as a pickle file
    with open(vocab_pkl_path, 'wb') as f:
        pickle.dump(vocab_dict, f)
    logging.info(f"Vocabulary saved to {vocab_pkl_path}")

    logging.debug(f"Saving vocabulary to text file: {vocab_txt_path}")
    # Save vocabulary as a text file with format "token value"
    with open(vocab_txt_path, 'w', encoding='utf-8') as f:
        for word, idx in sorted(vocab_dict.items(), key=lambda item: item[1]):
            f.write(f"{word} {idx}\n")
    logging.info(f"Vocabulary saved to {vocab_txt_path}")

# Define folder paths
current_dir = os.path.dirname(__file__)
dataset_folder = os.path.join(current_dir, '..', '..', 'datasets', 'DUC-2004-Dataset', 'DUC2004_Summarization_Documents', 'duc2004_testdata', 'tasks1and2')
output_folder = os.path.join(current_dir, '..', '..', 'preprocessed_data')

# Preprocess the entire dataset and save results
preprocessed_documents = preprocess_dataset(dataset_folder, output_folder)

# Output some results (first 3 documents as a sample)
for i, (file_path, preprocessed_doc) in enumerate(preprocessed_documents.items()):
    print(f"Preprocessed Document {i+1} ({file_path}): {preprocessed_doc[:10]}")  # Print first 10 tokens as a sample
    if i >= 2:
        break
