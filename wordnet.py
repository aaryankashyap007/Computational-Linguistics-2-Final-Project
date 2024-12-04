from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from clean import process_sentences
from read import read_sentences_to_list
from collections import Counter, deque
import math
import csv

def map_words_to_synsets(tokens):
    word_to_synset = {}
    for word in tokens:
        synsets = wn.synsets(word)
        if synsets:
            word_to_synset[word] = synsets[0]
    return word_to_synset

def count_frequencies(sentences):
    word_counts = Counter()
    synset_counts = Counter()
    total_words = 0
    
    for tokens in sentences:
        total_words += len(tokens)
        word_counts.update(tokens)
        
        # Map words to synsets and update synset counts
        word_to_synset = map_words_to_synsets(tokens)
        for synset in word_to_synset.values():
            synset_counts[synset] += 1
    
    return word_counts, synset_counts, total_words

def shortest_path_length(synset1, synset2):
    # Use BFS to find the shortest path
    queue = deque([(synset1, 0)])
    visited = set()
    
    while queue:
        current_synset, distance = queue.popleft()
        if current_synset == synset2:
            return distance
        visited.add(current_synset)
        neighbors = current_synset.hypernyms() + current_synset.hyponyms()
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append((neighbor, distance + 1))
    return float('inf')

def wu_palmer_similarity(synset1, synset2):
    """
    Compute the Wu-Palmer similarity between two synsets.
    If either synset has a depth of 0, return similarity as 0 to avoid ZeroDivisionError.
    """
    lcs = synset1.lowest_common_hypernyms(synset2)
    if not lcs:
        return 0
    depth_lcs = lcs[0].min_depth()
    depth1 = synset1.min_depth()
    depth2 = synset2.min_depth()

    # Handle division by zero
    if depth1 == 0 or depth2 == 0:
        return 0

    return (2 * depth_lcs) / (depth1 + depth2)

def calculate_ic(synset, synset_counts, total_words):
    freq = synset_counts.get(synset, 0)
    prob = freq / total_words
    return -math.log(prob) if prob > 0 else float('inf')

def generate_training_data(sentences, synset_counts, total_words):    
    for tokens in sentences:
        word_to_synset = map_words_to_synsets(tokens)
        words = list(word_to_synset.keys())
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words):
                if i < j:
                    synset1 = word_to_synset[word1]
                    synset2 = word_to_synset[word2]
                    path_sim = shortest_path_length(synset1, synset2)
                    wu_palmer_sim = wu_palmer_similarity(synset1, synset2)
                    ic_resnik = calculate_ic(synset1, synset_counts, total_words)
                    with open('training_data1.csv', 'a') as file:
                        file.write(f'{word1},{word2},{path_sim},{wu_palmer_sim},{ic_resnik}\n')

stop_words = list(set(stopwords.words('english')))
sentences_list = read_sentences_to_list('data.txt')
unique_sentences = set(sentences_list)
clean_sentences = tuple(process_sentences(unique_sentences))
tokenized_sentences = [sentence.lower().split() for sentence in clean_sentences]
tokenized_sentences_filtered = []
for sentence in tokenized_sentences:
    filtered_sentence = [token for token in sentence if token not in stop_words]
    tokenized_sentences_filtered.append(filtered_sentence)
    
word_counts, synset_counts, total_words = count_frequencies(tokenized_sentences_filtered)

filename = 'training_data2.csv'

with open(filename, 'w') as file:
    file.write("word1,word2,path_similarity,wu_palmer_similarity,ic_resnik\n")

generate_training_data(tokenized_sentences_filtered, synset_counts, total_words)

print(f"Training data has been saved to {filename}")