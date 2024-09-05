documents = [
    "Document one text.",
    "Document two text.",
    "Document three text.",
    # Add more documents here
]


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()  #Initialize the TF-IDF Vectorizer


tfidf_matrix = vectorizer.fit_transform(documents)  #Transform the documents sparse matrix

feature_names = vectorizer.get_feature_names_out() #Get the feature names


dense = tfidf_matrix.todense() #Convert the sparse matrix to dense matrix for easier readability


import pandas as pd

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names) #analyze or visualize the TF-IDF scores in a tabular format


print("tdf for first doc" .format(tfidf_df.iloc[0])) #tdf idf for first doc


# Inspect the DataFrame
print(tfidf_df)
