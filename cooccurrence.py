from clean import process_sentences
from read import read_sentences_to_list
from dict import tokens_dict, filter
from scipy.sparse import lil_matrix

def cooccurrence2(sentences, tokens_dictionary, cooccurence_matrix):
    for sentence in sentences:
        words = sentence.split()
        for i in range (len(words) - 2):
            token_i = words[i]
            token_j = words[i + 1]
            token_k = words[i + 2]
            idx_i = tokens_dictionary[token_i]
            idx_j = tokens_dictionary[token_j]
            idx_k = tokens_dictionary[token_k]
            cooccurence_matrix[idx_i, idx_j] += 1
            cooccurence_matrix[idx_j, idx_i] += 1
            cooccurence_matrix[idx_k, idx_i] += 1
            cooccurence_matrix[idx_i, idx_k] += 1

def cooccurrence4(sentences, tokens_dictionary, cooccurence_matrix):
    for sentence in sentences:
        words = sentence.split()
        for i in range (len(words) - 4):
            token_i = words[i]
            token_j = words[i + 1]
            token_k = words[i + 2]
            token_l = words[i + 3]
            token_m = words[i + 4]
            idx_i = tokens_dictionary[token_i]
            idx_j = tokens_dictionary[token_j]
            idx_k = tokens_dictionary[token_k]
            idx_l = tokens_dictionary[token_l]
            idx_m = tokens_dictionary[token_m]
            cooccurence_matrix[idx_i, idx_j] += 1
            cooccurence_matrix[idx_j, idx_i] += 1
            cooccurence_matrix[idx_i, idx_k] += 1
            cooccurence_matrix[idx_k, idx_i] += 1
            cooccurence_matrix[idx_i, idx_l] += 1
            cooccurence_matrix[idx_l, idx_i] += 1
            cooccurence_matrix[idx_i, idx_m] += 1
            cooccurence_matrix[idx_m, idx_i] += 1

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

def save_sparse_matrix(filename, sparse_matrix):
    with open(filename, 'w') as file:
        file.write(f'Row,Column,Value\n')
        for i, j in zip(*sparse_matrix.nonzero()):
            file.write(f"{i},{j},{sparse_matrix[i, j]}\n")

save_sparse_matrix('cooccurrence2.csv', cooccurrence_matrix2)
save_sparse_matrix('cooccurrence4.csv', cooccurrence_matrix4)

