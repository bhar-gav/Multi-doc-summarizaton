import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

os.environ['XDG_SESSION_TYPE'] = 'x11'   # ignore XDG_SESSION_TYPE'=wayland' warning/error in linux 

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

    print(f"Traversing directory: {directory_path}")
    # Traverse the directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    print(f"Reading file: {file_path}")
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        documents.append(f.read())
                        filenames.append(file_path)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    print(f"Read {len(documents)} documents from directory.")
    return documents, filenames

# Define the directory containing documents
directory_path = os.path.join(os.path.dirname(__file__), '..', '..','preprocessed_data','duc2004_tasks1and2_docs')

print("Starting document reading...")
# Read all documents
documents, filenames = read_documents_from_directory(directory_path)

print("Documents read. Starting TF-IDF Vectorization...")
# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

print("TF-IDF Vectorization completed. Computing cosine similarity matrix...")

# Compute cosine similarity matrix
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("Cosine similarity matrix computed. ")

# Print similarity matrix for inspection
print("\nCosine Similarity Matrix:")
print(np.round(cosine_sim_matrix, 2))


# Create a graph from the similarity matrix
G = nx.Graph()

print("\nCreating graph")

# Add nodes (sentences) with an attribute 'text'
for i, filename in enumerate(filenames):
    G.add_node(i, text=filename)

print(f"Adding edges to the graph. Number of sentences: {len(filenames)}")

# Add edges with weights based on cosine similarity
num_sentences = len(filenames)
for i in range(num_sentences):
    for j in range(i + 1, num_sentences):
        if cosine_sim_matrix[i, j] > 0:  # Only add edge if similarity > 0
            G.add_edge(i, j, weight=cosine_sim_matrix[i, j])

# Count the number of nodes and edges
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")

print("\nGraph creation completed. Visualizing graph...")

# Visualize the graph
pos = nx.spring_layout(G, seed=42,k=0.1)  # Layout for visualization

weights = nx.get_edge_attributes(G, 'weight')

# Create a larger figure
fig, ax = plt.subplots(figsize=(15, 15))  # Increase figure size for better visibility

nx.draw(G, pos, with_labels=True, node_size=20, node_color='skyblue', font_size=5, font_weight='bold')
nx.draw_networkx_edge_labels(G, pos, edge_labels={(i, j): f"{w:.2f}" for (i, j, w) in G.edges(data='weight')})

# Save the plot to a file
plt.savefig("results/input-graph/network.png", format='png', dpi=300)  # For PNG format
print("generated network.png")

plt.savefig("results/input-graph/network.pdf", format='pdf')  # For PDF format
print("generated network.pdf")

plt.savefig("results/input-graph/network.svg", format='svg')  # For SVG format
print("generated network.svg")
