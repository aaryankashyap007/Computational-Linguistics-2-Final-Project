def read_sentences_to_list(file_path):
    sentences = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            sentences = [line.strip() for line in file if line.strip()]
        return sentences
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
