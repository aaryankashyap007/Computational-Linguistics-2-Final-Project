from nltk.corpus import stopwords

stop_words = list(set(stopwords.words('english')))

def tokens_dict(sentences):
    words = []
    for sentence in sentences:
        words.extend(sentence.split())
    
    unique_words = list(set(words))
    unique_filtered_words = [word for word in unique_words if word not in stop_words]
    # print(unique_filtered_words)
    return {token: i for i, token in enumerate(unique_filtered_words)}
    # return {token: i for i, token in enumerate(unique_words)}

def filter(sentences):
    filtered_sentences = []
    for sentence in sentences:
        words = sentence.split()
        filtered_words = [word for word in words if word not in stop_words]
        filtered_sentence = ' '.join(filtered_words)
        filtered_sentences.append(filtered_sentence)
    return filtered_sentences