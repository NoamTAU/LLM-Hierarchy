import nltk

def load_filtered_treebank():
    """
    Load and filter the Treebank dataset.

    Returns:
        tuple: A tuple containing filtered words, filtered tags, and filtered tagged words.
    """
    # Download the necessary NLTK datasets
    nltk.download('treebank')
    nltk.download('brown')

    from nltk.corpus import treebank

    # Load the tagged words from the Treebank corpus
    words = treebank.tagged_words()[:-1]

    # Step 1: Identify and store the indices of the elements that need to be removed
    indices_to_remove = [i for i, (word, tag) in enumerate(words) if word.startswith('*') or word.endswith('*')]

    # Step 2: Remove the elements from the original list
    filtered_tagged_words = [word for i, word in enumerate(words) if i not in indices_to_remove]

    # Step 3: Extract the tags from the original list
    single_words = [word[0] for word in words]
    tags = [word[1] for word in words]

    # Step 4: Remove the corresponding tags using the stored indices
    filtered_tags = [tag for i, tag in enumerate(tags) if i not in indices_to_remove]
    filtered_words = [word for i, word in enumerate(single_words) if i not in indices_to_remove]

    return filtered_words, filtered_tags, filtered_tagged_words

def split_into_sentences(words, chunk_size=100):
    """
    Split a list of words into chunks of a specified size.

    Args:
        words (list): A list of words to split.
        chunk_size (int): The size of each chunk.

    Returns:
        list: A list of sentences, where each sentence is a chunk of words.
    """
    sentences = []
    for i in range(0, len(words), chunk_size):
        sentences.append(words[i:i + chunk_size])
    return sentences



#### Example usage ####
# Import the functions from the treebank_utils module
# from treebank_utils import load_filtered_treebank, split_into_sentences

# # Load the filtered Treebank dataset
# filtered_words, filtered_tags, filtered_tagged_words = load_filtered_treebank()

# # Split the filtered words into sentences of 100 words each
# sentences = split_into_sentences(filtered_words)

# # Print the sentences
# for i, sentence in enumerate(sentences[:2]):
#     print(f"Sentence {i + 1}: {' '.join(sentence)}")
