import pandas as pd
import csv
from collections import Counter
from nltk.corpus import wordnet as wn
import numpy as np
from functools import lru_cache

# Load the training data
csv_file = 'training_data1.csv'
training_data = pd.read_csv(csv_file)

# Create a dictionary for quick lookup of precomputed training data
training_dict = {
    (row['word1'], row['word2']): (row['path_similarity'], row['wu_palmer_similarity'], row['ic_resnik'])
    for _, row in training_data.iterrows()
}

# Calculate word and synset counts from the training data
def calculate_word_and_synset_counts(data):
    words = list(data['word1']) + list(data['word2'])
    word_counts = Counter(words)
    total_unique_words = len(word_counts)
    return total_unique_words, word_counts

total_unique_words, synset_counts = calculate_word_and_synset_counts(training_data)

# Helper function to calculate similarity metrics
def calculate_similarity_metrics(word1, word2):
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)

    if not synsets1 or not synsets2:
        print(f"No synsets found for '{word1}' or '{word2}'.")
        return [float('inf'), 0, float('inf')]

    # Iterate over all combinations of synsets and calculate metrics
    path_similarities = []
    wu_palmer_similarities = []
    ic_resniks = []

    for s1 in synsets1:
        for s2 in synsets2:
            path_similarities.append(shortest_path_length(s1, s2))
            wu_palmer_similarities.append(wu_palmer_similarity(s1, s2))
            ic_resniks.append(calculate_ic(s1, synset_counts, total_unique_words))
    
    # Return the best scores
    path_similarity = min(path_similarities) if path_similarities else float('inf')
    wu_palmer_sim = max(wu_palmer_similarities) if wu_palmer_similarities else 0
    ic_resnik = max(ic_resniks) if ic_resniks else float('inf')

    return [path_similarity, wu_palmer_sim, ic_resnik]

# Compute shortest path length using WordNet's built-in method
@lru_cache(maxsize=None)
def shortest_path_length(s1, s2):
    try:
        return s1.shortest_path_distance(s2) or float('inf')
    except:
        return float('inf')

# Compute Wu-Palmer similarity
def wu_palmer_similarity(s1, s2):
    lcs = s1.lowest_common_hypernyms(s2)
    if not lcs:
        return 0
    depth_lcs = lcs[0].min_depth()
    depth1 = s1.min_depth()
    depth2 = s2.min_depth()
    if depth1 == 0 or depth2 == 0:
        return 0
    return (2 * depth_lcs) / (depth1 + depth2)

# Compute information content (IC) for Resnik similarity
def calculate_ic(synset, counts, total):
    freq = counts.get(synset, 0)
    prob = freq / total
    return -np.log(prob) if prob > 0 else float('inf')

# Predict word relatedness
def predict_word_relatedness(word1, word2):
    key = (word1, word2) if (word1, word2) in training_dict else (word2, word1)
    if key in training_dict:
        path_sim, wu_palmer_sim, ic_resnik = training_dict[key]
        return (path_sim + wu_palmer_sim + ic_resnik) / 3

    metrics = calculate_similarity_metrics(word1, word2)
    avg_similarity = sum(metrics) / len(metrics)
    
    # Handle extreme cases
    if avg_similarity == float('inf') or avg_similarity == 0:
        avg_similarity = 0.1  # Fallback value
    
    return avg_similarity

# Save predictions to a CSV file
def save_predictions(word1, word2, score, filename='new_predictions.csv'):
    with open(filename, 'a') as f:
        f.write(f"{word1},{word2},{score}\n")

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

# Example usage
if __name__ == "__main__":
    # word1 = "order"
    # word2 = "request"
    filename = "test_word_pairs.csv"
    word_pairs = read_word_pairs_from_csv(filename)
    with open('new_predictions.csv', 'w') as file:
        file.write(f'Word1,Word2,Similarity\n')
    for word1, word2 in word_pairs:
        relatedness_score = predict_word_relatedness(word1, word2)
        # print(f"Relatedness between '{word1}' and '{word2}': {relatedness_score}")
        save_predictions(word1, word2, relatedness_score)
    