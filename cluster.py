from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import numpy as np

fileObj = open('content.txt', "r")  # opens the file in read mode
words = fileObj.read().splitlines()

n_grams_to_use = []
n_grams_to_use.extend(words)

def split_nGrams(n_grams_to_use):
    ngrams_splited = [each.split() for each in n_grams_to_use]
    return ngrams_splited
ngrams_splited = split_nGrams(n_grams_to_use)
len(ngrams_splited)


def average_word_vectors(list_words, model, vocabulary, num_features):
    """
    This function will take each tokenized sentence having bigrams or trigrams, 
    model = the mapping_of_word_to_vector dictionary, vocabulary = unique set of keys(words) present in model,
    num_features = 50
    
    This function will return the average of feature vector for each word present in list_words.
    """
    # Created array of zeros (type float) of size num_features, i.e., 50.
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0.
    # Put it in try block so that if any exception occur, it will be dealt by below exception block.
    try:
        # Check if word is in passed list_of_words or not.
        for word in list_words:
            # Check if word is in general vocabulary or not (the unique set of words in word embedding).
            if word in vocabulary:
                # Increment number_of_words
                nwords = nwords + 1
                # add vector array of corresponding key in model which matches the passed word.
                feature_vector = np.add(feature_vector, model[word])

        if nwords:
            # Take average of feature_vector by dividing with total number of words
            feature_vector = np.divide(feature_vector, nwords)

        return feature_vector

    except:
        # If the exception occurs, while the word isn't found in vocabulary, it will return the array of zeros
        return np.zeros((num_features,), dtype="float64")


def averaged_word_vectorizer(corpus, model, num_features):
    """
    This function is taking corpus of bigrams & trigrams, w2v mappings, num of features as a input arguments.
    and returning array of features after taking average using average_word_vectors() function.
    """
    # Get the unique keys out of word_to_vector_map dictionary.
    vocabulary = set(model.keys())
    # Call function average_word_vectors which is returning with averaged vectors for each word in tokenized sentence.
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                for tokenized_sentence in ngrams_splited]
    return np.array(features)


def read_glove(glove_path):
    """
    This function will read glove data from text file and do the following:
    1. prepare dictionary of words and vectors
    2. prepare dictionary of words and index
    3. prepare dictionary of index and words
    """
    # Read word_embedding file stored on glove_path specified.
    with open(glove_path, 'r', encoding='utf-8')as inp_file:

        words = set()
        word_to_vec_map = {}

        # For every line in embedding file which contains the word & the corresponding vector.
        for line in inp_file:
            # convert each line in embedding file to a list of elements.
            line = line.strip().split()
            # Get first element of the list, i.e., word of each list.
            curr_word = line[0]
            # Add the distinct set of words.
            words.add(curr_word)
            # Create dictionary that will map current word to that of it's vector representation.
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        # For every word in sorted dictionary of words
        for w in sorted(words):
            # map index to each words
            words_to_index[w] = i
            # map words to each index
            index_to_words[i] = w
            i += 1

        return words_to_index, index_to_words, word_to_vec_map


# load glove vectors from pre-trained model domain dataset
glove_path = r"Generating_nGrams\Text Clustering\domain_embeddings.txt"
new_words_to_index, new_index_to_words, new_word_to_vec_map = read_glove(
    glove_path)
