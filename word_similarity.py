import csv
from cooccurrence_similarity import cosine, cooccurrence2, cooccurrence4
from wordnet_similarity import predict_word_relatedness
from clean import process_sentences
from read import read_sentences_to_list
from dict import tokens_dict, filter
from framenet_similarity import batch_calculate_similarity
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import svds

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



if __name__ == "__main__":
    filename = "test_word_pairs.csv"
    word_pairs = read_word_pairs_from_csv(filename)
    print(f"Total word pairs: {len(word_pairs)}")
    # print(word_pairs)
    
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

    n_components = 100

    cooccurrence_matrix2 = cooccurrence_matrix2.tocsr()
    U, S, VT = svds(cooccurrence_matrix2, k = n_components)

    word_embeddings2 = U * S

    cooccurrence_matrix4 = cooccurrence_matrix4.tocsr()
    U, S, VT = svds(cooccurrence_matrix4, k = n_components)

    word_embeddings4 = U * S

    with open('result.txt', 'w') as file:
        i = 0
        for word1, word2 in word_pairs:
            file.write(f'''{i}.\nWord 1: {word1}\n
                       Word 2: {word2}\n
                       Cosine Similarity (Window Size 2): {cosine(word_embeddings2, word_pairs, tokens_dictionary)}\n
                       Cosine Similarity (Window Size 4): {cosine(word_embeddings4, word_pairs, tokens_dictionary)}\n
                       Wordnet Similarity: {predict_word_relatedness(word1, word2)}\n
                       Framenet Similarity: {batch_calculate_similarity(word_pairs)}\n\n''')
            i += 1