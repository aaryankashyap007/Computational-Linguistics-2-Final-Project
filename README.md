# Computational-Linguistics-2-Final-Project - Word Similarity Project

## Overview
This project aims to compute semantic similarity between words using various methods, including co-occurrence matrices, WordNet, and FrameNet. The project is structured into several Python modules, each responsible for specific tasks such as data cleaning, co-occurrence calculation, similarity computation, and data generation.

---

## Modules

### 1. `clean.py`
This module contains functions to preprocess sentences by:
- Converting sentences to lowercase.
- Removing non-alphabetic characters.
- Filtering out stop words.

**Key Function**:
- `process_sentences(sentences)`: Cleans and processes a list of sentences.

---

### 2. `cooccurrence_similarity.py`
This module calculates co-occurrence similarity between word pairs using co-occurrence matrices. It reads word pairs from a CSV file and computes cosine similarity based on two different window sizes (2 and 4).

**Key Functions**:
- `read_word_pairs_from_csv(filename)`: Reads word pairs from a CSV file.
- `cosine_similarity(vec1, vec2)`: Computes cosine similarity between two vectors.
- `cosine2(word_embeddings)` and `cosine4(word_embeddings)`: Calculate cosine similarity for word pairs using embeddings from co-occurrence matrices.

---

### 3. `cooccurrence.py`
This module constructs co-occurrence matrices based on the cleaned sentences. It defines two functions for calculating co-occurrences based on different window sizes.

**Key Functions**:
- `cooccurrence2(sentences, tokens_dictionary, cooccurrence_matrix)`: Calculates co-occurrences for a window size of 2.
- `cooccurrence4(sentences, tokens_dictionary, cooccurrence_matrix)`: Calculates co-occurrences for a window size of 4.
- `save_sparse_matrix(filename, sparse_matrix)`: Saves the co-occurrence matrices to CSV files.

---

### 4. `dict.py`
This module provides functions to create a dictionary of tokens and filter sentences based on stop words.

**Key Functions**:
- `tokens_dict(sentences)`: Creates a dictionary mapping unique words to indices.
- `filter(sentences)`: Filters out stop words from sentences.

---

### 5. `framenet_similarity.py`
This module computes similarity based on FrameNet data. It loads training data and defines functions to calculate frame overlap and distance between word pairs.

**Key Functions**:
- `calculate_similarity(word1, word2)`: Computes similarity based on frame overlap and distance.
- `batch_calculate_similarity(word_pairs)`: Computes similarity for a batch of word pairs.

---

### 6. `framenet.py`
This module maps words to frames using the FrameNet corpus and generates training data for frame-based similarity.

**Key Functions**:
- `map_words_to_frames(tokens)`: Maps words to their corresponding frames.
- `generate_training_data(sentences, word_to_frames_map)`: Generates training data for frame-based similarity.

---

### 7. `read.py`
This module provides a function to read sentences from a text file into a list.

**Key Function**:
- `read_sentences_to_list(file_path)`: Reads sentences from a specified file.

---

### 8. `test.py`
This module generates word pairs with semantic relations (synonyms, antonyms, hypernyms, hyponyms) using WordNet and saves them to a CSV file.

**Key Functions**:
- `generate_word_pairs()`: Generates pairs of words with various semantic relations.
- `save_word_pairs_to_csv(word_pairs, filename)`: Saves generated word pairs to a CSV file.

---

### 9. `word_similarity.py`
This module integrates various similarity measures (co-occurrence, WordNet, FrameNet) and outputs the results to a file.

**Key Functions**:
- `read_word_pairs_from_csv(filename)`: Reads word pairs from a CSV file.

---

### 10. `wordnet_similarity.py`
This module computes similarity metrics using WordNet, including path similarity, Wu-Palmer similarity, and information content (IC) for Resnik similarity.

**Key Functions**:
- `calculate_similarity_metrics(word1, word2)`: Computes various similarity metrics for two words.
- `predict_word_relatedness(word1, word2)`: Predicts relatedness between two words based on precomputed training data.

---

### 11. `wordnet.py`
This module provides functions to map words to synsets and calculate various similarity metrics based on WordNet.

**Key Functions**:
- `map_words_to_synsets(tokens)`: Maps words to their corresponding synsets.
- `generate_training_data(sentences, synset_counts, total_words)`: Generates training data for WordNet-based similarity.

---

## Usage

### Data Preparation
1. Ensure you have a text file (`data.txt`) containing sentences for processing.
2. Run `test.py` to generate a CSV file (`test_word_pairs.csv`) with word pairs.

### Compute Similarities
Run `word_similarity.py` to compute similarities between the generated word pairs using various methods and save the results to a file (`result.txt`).

---

## Requirements

- Python 3.x
- NLTK library (for natural language processing tasks)
- Pandas library (for data manipulation)
- SciPy library (for scientific computing)

### Installation
Install the required libraries using pip:

```bash
pip install nltk pandas scipy
```

### NLTK Data
Download the necessary NLTK resources:

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
```

---

## Example

### Steps
1. Place your sentences in `data.txt`.
2. Run the following command to generate word pairs:
   ```bash
   python test.py
   ```
3. Compute the similarities:
   ```bash
   python word_similarity.py
   ```

### Output
The results will be saved in `result.txt`, containing:
- Cosine similarities from co-occurrence matrices.
- WordNet similarities.
- FrameNet similarities for each word pair.

---

## Conclusion
This project provides a comprehensive framework for computing semantic similarity between words using multiple approaches. You can extend the functionality by adding more similarity measures or improving the existing algorithms.
