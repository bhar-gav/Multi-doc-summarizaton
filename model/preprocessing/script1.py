import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

<<<<<<< HEAD
# Download NLTK data files (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
=======
# # Download NLTK data files (if not already downloaded)
# nltk.download('punkt')
# nltk.download('stopwords')
>>>>>>> be228c1d0 (update)

# Initialize the stemmer and stop words list
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function for Preprocessing a Single Document
def preprocess_document(document):
    """
    Preprocess a single document: lowercase, remove special characters, tokenize, remove stopwords, and stem.
    """
    document = document.lower()
    document = re.sub(r'[^a-z\s]', '', document)
    words = word_tokenize(document)
    processed_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return processed_words

# Function to Load and Preprocess Documents from Dataset Folder
def preprocess_dataset(dataset_folder, output_folder):
    """
    Process all documents in dataset_folder and save preprocessed documents to output_folder.
    """
    preprocessed_data = {}
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    logging.info(f"Starting preprocessing of files in {dataset_folder}")
    
    # Traverse through each folder in the dataset directory
    for root, dirs, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith(".txt"):  # Process only .txt files
                file_path = os.path.join(root, file)
                
                try:
                    logging.info(f"Processing file: {file_path}")
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        document = f.read()
                    
                    preprocessed_doc = preprocess_document(document)
                    
                    # Construct the output file path, preserving the directory structure
                    relative_path = os.path.relpath(file_path, dataset_folder)
                    output_file_path = os.path.join(output_folder, relative_path)
                    
                    # Ensure output directory exists
                    output_file_dir = os.path.dirname(output_file_path)
                    if not os.path.exists(output_file_dir):
                        os.makedirs(output_file_dir)
                    
                    # Save the preprocessed document
                    with open(output_file_path, 'w', encoding='utf-8') as out_file:
                        out_file.write(' '.join(preprocessed_doc))
                    
                    preprocessed_data[file_path] = preprocessed_doc
                    logging.info(f"Processed and saved: {output_file_path}")

                except Exception as e:
                    logging.error(f"Error processing file {file_path}: {e}")

    logging.info(f"Preprocessing completed. Processed files are saved in {output_folder}.")
    return preprocessed_data


# Define folder paths
dataset_folder = '/home/bhargav/Desktop/Bhargav/projects/Major_Project/Multi-doc-summarization/datasets/DUC-2004-Dataset/DUC2004_Summarization_Documents/duc2004_testdata/tasks1and2'
output_folder = '/home/bhargav/Desktop/Bhargav/projects/Major_Project/Multi-doc-summarization/preprocessed_data'

# Preprocess the entire dataset and save results
preprocessed_documents = preprocess_dataset(dataset_folder, output_folder)

# Output some results (first 3 documents as a sample)
for i, (file_path, preprocessed_doc) in enumerate(preprocessed_documents.items()):
    print(f"Preprocessed Document {i+1} ({file_path}): {preprocessed_doc[:10]}")  # Print first 10 tokens as a sample
    if i >= 2:  
        break
