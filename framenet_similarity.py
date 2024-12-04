import pandas as pd
import csv

# Load the Framenet training data
csv_file = 'framenet_training_data.csv'  # Replace with your actual file path
training_data = pd.read_csv(csv_file)

# Preprocess training data into a dictionary for faster lookups
training_dict = {
    (row['word1'], row['word2']): (row['frame_overlap'], row['frame_distance'])
    for _, row in training_data.iterrows()
}

# Define a similarity metric
def calculate_similarity(word1, word2, epsilon=1e-6, default_similarity=0):
    # Search for the word pair in the dictionary
    key = (word1, word2) if (word1, word2) in training_dict else (word2, word1)
    if key in training_dict:
        frame_overlap, frame_distance = training_dict[key]
        
        # Handle cases where frame_distance is infinity
        if frame_distance == float('inf'):
            return frame_overlap  # Use only overlap when distance is infinity
        
        # Calculate similarity
        similarity = frame_overlap - (1 / (frame_distance + epsilon))
        return similarity
    
    # If the word pair is not found in the training data
    return default_similarity

# Batch similarity function
def batch_calculate_similarity(word_pairs):
    results = []
    for word1, word2 in word_pairs:
        similarity = calculate_similarity(word1, word2)
        results.append((word1, word2, similarity))
    return results

# Save results to a CSV file
def save_results(word1, word2, similarity, filename='framenet_results.txt'):
    with open(filename, 'a') as f:
        f.write(f"Similarity between '{word1}' and '{word2}': {similarity}\n")

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
    filename = "test_word_pairs.csv"
    word_pairs = read_word_pairs_from_csv(filename)
    
    for word1, word2 in word_pairs:
        similarity_score = calculate_similarity(word1, word2)
        # print(f"Similarity between '{word1}' and '{word2}': {similarity_score}")
        
        # Save result
        save_results(word1, word2, similarity_score)

        # Batch example
        # word_pairs = [("right", "possession"), ("credit", "cards"), ("plant", "genus")]
        batch_results = batch_calculate_similarity(word_pairs)
        # with open('framenet_results', 'w') as file:
        #     for word1, word2, score in batch_results:
        #         file.write(f"Similarity between '{word1}' and '{word2}': {score}\n")
