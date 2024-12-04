from cooccurrence import cooccurrence2, cooccurrence4
from clean import process_sentences
from read import read_sentences_to_list
from dict import tokens_dict, filter
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import svds
import csv

def read_word_pairs_from_csv(filename):
    """
    Reads a CSV file containing word pairs and returns a list of (word1, word2) tuples.

    Args:
        filename (str): Path to the CSV file.

    Returns:
        list: A list of tuples where each tuple contains (word1, word2).
    """
    word_pairs = []

    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            word1 = row['Word1']
            word2 = row['Word2']
            word_pairs.append((word1, word2))

    return word_pairs

sentences_list = read_sentences_to_list('data.txt')
unique_sentences = set(sentences_list)
clean_sentences = process_sentences(unique_sentences)
clean_filtered_sentences = filter(clean_sentences)
tokens_dictionary = tokens_dict(clean_sentences)
length = len(tokens_dictionary)

cooccurrence_matrix2 = lil_matrix((length, length), dtype=float)
cooccurrence_matrix4 = lil_matrix((length, length), dtype=float)

cooccurrence2(clean_filtered_sentences, tokens_dictionary, cooccurrence_matrix2)
cooccurrence4(clean_filtered_sentences, tokens_dictionary, cooccurrence_matrix4)

filename = "test_word_pairs.csv"
word_pairs = read_word_pairs_from_csv(filename)
n_components = 100

cooccurrence_matrix2 = cooccurrence_matrix2.tocsr()
U, S, VT = svds(cooccurrence_matrix2, k = n_components)

word_embeddings2 = U * S

cooccurrence_matrix4 = cooccurrence_matrix4.tocsr()
U, S, VT = svds(cooccurrence_matrix4, k = n_components)

word_embeddings4 = U * S

def cosine_similarity(vec1, vec2):

    dot_product = np.dot(vec1, vec2)
    
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    similarity = dot_product / (norm_vec1 * norm_vec2)
    
    return similarity

def cosine2(word_embeddings):
    i = 1  # Counter for result numbering
    for word1, word2 in word_pairs:
        if word1 not in tokens_dictionary or word2 not in tokens_dictionary:
            # Log missing words and skip the pair
            print(f"Skipping pair ('{word1}', '{word2}') - Missing in dictionary.")
            continue
        
        # Retrieve indices
        idx1, idx2 = tokens_dictionary[word1], tokens_dictionary[word2]
        embedding1, embedding2 = word_embeddings[idx1], word_embeddings[idx2]

        # Compute similarity
        similarity = cosine_similarity(embedding1, embedding2)

        # Write results to file
        with open('cooccurrence_result1.txt', 'a') as file:
            file.write(f"""{i}.
Word 1: {word1}
Word 2: {word2}
Cosine Similarity: {similarity:.4f}\n\n""")
        i += 1


def cosine4(word_embeddings):
    i = 1  # Counter for result numbering
    for word1, word2 in word_pairs:
        if word1 not in tokens_dictionary or word2 not in tokens_dictionary:
            # Log missing words and skip the pair
            print(f"Skipping pair ('{word1}', '{word2}') - Missing in dictionary.")
            continue
        
        # Retrieve indices
        idx1, idx2 = tokens_dictionary[word1], tokens_dictionary[word2]
        embedding1, embedding2 = word_embeddings[idx1], word_embeddings[idx2]

        # Compute similarity
        similarity = cosine_similarity(embedding1, embedding2)

        # Write results to file
        with open('cooccurrence_result2.txt', 'a') as file:
            file.write(f"""{i}.
Word 1: {word1}
Word 2: {word2}
Cosine Similarity: {similarity:.4f}\n\n""")
        i += 1

if __name__ == "__main__":
    cosine2(word_embeddings2)
    cosine4(word_embeddings4)