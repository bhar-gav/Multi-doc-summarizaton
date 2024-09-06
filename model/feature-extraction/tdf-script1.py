import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def read_documents_from_directory(directory_path):
    """
    Reads all text files from the given directory path and its subdirectories.
    
    Args:
        directory_path (str): The root directory containing subdirectories and text files.

    Returns:
        tuple: A tuple where the first element is a list of document texts and 
               the second element is a list of file paths.
    """
    documents = []
    filenames = []

    # Traverse the directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        documents.append(f.read())
                        filenames.append(file_path)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    
    return documents, filenames

# Define the root directory containing documents
directory_path = os.path.join(os.path.dirname(__file__), '..', '..','preprocessed_data','duc2004_tasks1and2_docs')
 

# Read all documents
documents, filenames = read_documents_from_directory(directory_path)

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Transform the documents into a TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(documents)

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Convert the sparse matrix to a dense matrix for readability
dense = tfidf_matrix.todense()

# Create a DataFrame from the dense matrix
tfidf_df = pd.DataFrame(dense, columns=feature_names, index=filenames)

# Print TF-IDF for the first document
print("TF-IDF for the first document:")
print(tfidf_df.iloc[0])

# Inspect the DataFrame
print("\nTF-IDF DataFrame:")
print(tfidf_df)
