import re

def process_sentences(sentences):
    processed_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub(r'[^a-z\s]', '', sentence)
        processed_sentences.append(sentence)
    return processed_sentences
