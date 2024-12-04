import csv
import random
from nltk.corpus import wordnet as wn

# Ensure NLTK resources are downloaded
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

# Helper function to get synonyms
def get_synonyms(word):
    synsets = wn.synsets(word)
    synonyms = set()
    for synset in synsets:
        for lemma in synset.lemmas():
            if lemma.name().lower() != word.lower():
                synonyms.add(lemma.name().lower())
    return list(synonyms)

# Helper function to get antonyms
def get_antonyms(word):
    synsets = wn.synsets(word)
    antonyms = set()
    for synset in synsets:
        for lemma in synset.lemmas():
            for antonym in lemma.antonyms():
                antonyms.add(antonym.name().lower())
    return list(antonyms)

# Helper function to get hypernyms
def get_hypernyms(word):
    synsets = wn.synsets(word)
    hypernyms = set()
    for synset in synsets:
        for hypernym in synset.hypernyms():
            hypernyms.add(hypernym.lemmas()[0].name().lower())
    return list(hypernyms)

# Helper function to get hyponyms
def get_hyponyms(word):
    synsets = wn.synsets(word)
    hyponyms = set()
    for synset in synsets:
        for hyponym in synset.hyponyms():
            hyponyms.add(hyponym.lemmas()[0].name().lower())
    return list(hyponyms)

# Helper function to generate random words
def get_random_words(word_count=100):
    words = set()
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            words.add(lemma.name().lower())
            if len(words) >= word_count:
                return list(words)
    return list(words)

# Generate pairs of words with semantic relations
def generate_word_pairs():
    word_pairs = []
    seen_pairs = set()

    # Collect a variety of words from WordNet
    words = get_random_words(500)  # Increase to 500 base words for variety

    for word in words:
        # Synonyms
        synonyms = get_synonyms(word)
        for synonym in random.sample(synonyms, min(5, len(synonyms))):  # Increase to 5
            pair = tuple(sorted((word, synonym)))
            if pair not in seen_pairs:
                word_pairs.append((word, synonym, 'synonym'))
                seen_pairs.add(pair)

        # Antonyms
        antonyms = get_antonyms(word)
        for antonym in random.sample(antonyms, min(5, len(antonyms))):  # Increase to 5
            pair = tuple(sorted((word, antonym)))
            if pair not in seen_pairs:
                word_pairs.append((word, antonym, 'antonym'))
                seen_pairs.add(pair)

        # Hypernyms
        hypernyms = get_hypernyms(word)
        for hypernym in random.sample(hypernyms, min(5, len(hypernyms))):  # Increase to 5
            pair = tuple(sorted((word, hypernym)))
            if pair not in seen_pairs:
                word_pairs.append((word, hypernym, 'hypernym'))
                seen_pairs.add(pair)

        # Hyponyms
        hyponyms = get_hyponyms(word)
        for hyponym in random.sample(hyponyms, min(5, len(hyponyms))):  # Increase to 5
            pair = tuple(sorted((word, hyponym)))
            if pair not in seen_pairs:
                word_pairs.append((word, hyponym, 'hyponym'))
                seen_pairs.add(pair)

    # Add random unrelated word pairs to reach 1,000 pairs
    random_words = get_random_words(1000)
    while len(word_pairs) < 1000:
        word1, word2 = random.sample(random_words, 2)
        pair = tuple(sorted((word1, word2)))
        if pair not in seen_pairs:
            word_pairs.append((word1, word2, 'random'))
            seen_pairs.add(pair)

    # Shuffle the pairs
    random.shuffle(word_pairs)
    return word_pairs

# Save pairs to a CSV file
def save_word_pairs_to_csv(word_pairs, filename="test_word_pairs.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Word1", "Word2", "Relation"])  # Header
        writer.writerows(word_pairs)

# Main function to generate and save word pairs
if __name__ == "__main__":
    print("Generating word pairs...")
    word_pairs = generate_word_pairs()
    print(f"Generated {len(word_pairs)} word pairs.")
    save_word_pairs_to_csv(word_pairs)
    print(f"Word pairs saved to 'test_word_pairs.csv'.")