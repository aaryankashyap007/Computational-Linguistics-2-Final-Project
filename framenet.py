from nltk.corpus import framenet as fn
from nltk.corpus import stopwords
from collections import defaultdict, deque
from read import read_sentences_to_list
import csv
import re

stop_words = list(set(stopwords.words('english')))

def process_sentences(sentences):
    return [sentence.strip().lower().split() for sentence in sentences if sentence.strip()]

def map_words_to_frames(tokens):
    word_to_frames = defaultdict(list)
    for word in tokens:
        if word not in stop_words:
            escaped_word = re.escape(word)
            lus = fn.lus(name=escaped_word)
            for lu in lus:
                word_to_frames[word].append(lu.frame.name)
    return word_to_frames

def frame_overlap_similarity(frames1, frames2):
    common_frames = set(frames1) & set(frames2)
    return len(common_frames) / max(len(frames1), len(frames2)) if frames1 and frames2 else 0

def frame_distance(frames1, frames2):
    def bfs_distance(frame1, frame2):
        queue = deque([(frame1, 0)])
        visited = set()

        while queue:
            current_frame, distance = queue.popleft()
            if current_frame == frame2:
                return distance
            visited.add(current_frame)

            related_frames = []
            try:
                related_frames += fn.frame_by_name(current_frame).related()
            except KeyError:
                continue

            for neighbor in related_frames:
                if neighbor not in visited:
                    queue.append((neighbor.name, distance + 1))
        return float('inf')

    distances = [bfs_distance(f1, f2) for f1 in frames1 for f2 in frames2]
    return min(distances) if distances else float('inf')

def generate_training_data(sentences, word_to_frames_map):

    for tokens in sentences:
        word_to_frames = {word: word_to_frames_map[word] for word in tokens if word in word_to_frames_map}
        words = list(word_to_frames.keys())

        for i, word1 in enumerate(words):
            for j in range(i + 1, len(words)):
                word2 = words[j]
                
                if word1 not in stop_words and word2 not in stop_words:
                
                    frames1 = word_to_frames[word1]
                    frames2 = word_to_frames[word2]

                    overlap_sim = frame_overlap_similarity(frames1, frames2)
                    distance_sim = frame_distance(frames1, frames2)

                    with open('framenet_training_data.csv', 'a') as file:
                        file.write(f'{word1},{word2},{overlap_sim},{distance_sim}\n')

if __name__ == "__main__":
    sentences_list = read_sentences_to_list('data.txt')
    unique_sentences = set(sentences_list)

    tokenized_sentences = process_sentences(unique_sentences)

    one_percent_size = max(1, int(len(tokenized_sentences) * 0.5))
    subset = tokenized_sentences[:one_percent_size]
    
    print("Mapping words to frames...")
    word_to_frames_map = {}
    for tokens in subset:
        word_to_frames_map.update(map_words_to_frames(tokens))
    
    with open('framenet_training_data.csv', 'w') as file:
        file.write('word1,word2,frame_overlap,frame_distance\n')
    print("Generating training data...")
    generate_training_data(subset, word_to_frames_map)

    print(f"Training data has been saved")
