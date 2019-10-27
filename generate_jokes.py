import gensim
import pprint
import random
import spacy
import string

from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
from gensim import models

from sklearn.feature_extraction import DictVectorizer

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Load spacy module
nlp = spacy.load("en")


def read_jokes(file):
    """
        Reads jokes from file and separates setup and punchline.
        Compiles jokes to dictionary with setup NOUN as key.
        Dictionary contains sentence strings and lemma lists.
        setup = question
        punchline = answer
    """
    # Read jokes from file, separate setup + punchline
    jokes = {}
    with open(file, "r") as f:
        for line in f:
            sents = sent_tokenize(line)

            # Check that joke contains two sentences and the first sentence ends in "?"
            if len(sents) == 2 and sents[0][len(sents[0])-1] == "?":
                que_tokens = nlp(sents[0])
                ans_tokens = nlp(sents[1])

                # Find keyword, has to be NOUN
                keyword = ""
                for token in que_tokens:
                    if not keyword and token.pos_ == "NOUN":
                        keyword = token.lemma_

                # Remove stopwords and punctuation
                que_lemmas = [token.lemma_ for token in que_tokens if not token.is_stop and token.is_alpha and token.lemma_ != "-PRON-"]
                ans_lemmas = [token.lemma_ for token in ans_tokens if not token.is_stop and token.is_alpha and token.lemma_ != "-PRON-"]

                # Dictionary key is a noun lemma from question
                jokes[keyword] = {"QUESTION":[token.text for token in que_tokens],  # question as list of tokens (word forms)
                                  "ANSWER":[token.text for token in ans_tokens],    # answer   as list of tokens (word forms)
                                  "QUESTION_LEMMAS":que_lemmas,                     # question as list of lemmas
                                  "ANSWER_LEMMAS":ans_lemmas}                       # answer   as list of lemmas
    return jokes


# Extracts features from token
def get_features(token, i, sent):
    token_feature = {
        "TOKEN"         : token.text,       # Token itself
        "FIRST"         : i == 0,           # Is token at the beginning of the sentence
        "LAST"          : i == len(sent)-1, # Is token at the end of the sentence

        "CAPITALIZED"   : token.text[0].upper() == token.text[0],     # Is first letter of token a capital letter
        "CAP_ALL"       : token.text.upper() == token.text,           # Are all letters of token capital letters
        "CAP_INSIDE"    : token.text[1:].lower() != token.text[1:],   # Is there any capital letters in the token
        "NUMERIC"       : any(char.isdigit() for char in token.text), # Is there any digits in the token
        
        "TOKEN_PREV"    : '' if i == 0 else sent[i - 1].text, # Previous token in the sentence
        "TOKEN_PREV_2"  : '' if i <= 1 else sent[i - 2].text, # Two previous tokens in the sentence

        "TOKEN_NEXT"    : '' if i == len(sent) - 1 else sent[i + 1].text, # Next token in the sentence
        "TOKEN_NEXT_2"  : '' if i >= len(sent) - 2 else sent[i + 2].text, # Two next tokens in the sentence

        "POS"           : token.pos_, # POS of token
        "TAG"           : token.tag_, # TAG of token
        "DEP"           : token.dep_, # DEP of token

        "POS_PREV"      : '' if i == 0 else sent[i - 1].pos_, # POS of previous token
        "TAG_PREV"      : '' if i == 0 else sent[i - 1].tag_, # TAG of previous token
        "DEP_PREV"      : '' if i == 0 else sent[i - 1].dep_, # DEP of previous token

        "POS_NEXT"      : '' if i == len(sent) - 1 else sent[i + 1].pos_, # POS of next token
        "TAG_NEXT"      : '' if i == len(sent) - 1 else sent[i + 1].tag_, # TAG of next token
        "DEP_NEXT"      : '' if i == len(sent) - 1 else sent[i + 1].dep_, # DEP of next token
    }
    return token_feature


# Transforms features into vectors
def vectorize_features(features):
    vectorizer = DictVectorizer(sparse=False)
    return vectorizer.fit_transform(features)


# Transforms feature vectors into gensim's corpus format
def vectors_to_corpus(features):
    return gensim.matutils.Dense2Corpus(features, documents_columns=False)


# Makes a corpus that is used to build the id dictionary.
def process_corpus(jokes_dict):
    processed_corpus = []

    # Combine question lemma lists and answer lemma lists
    for key in jokes_dict:
        question = [token for token in jokes_dict[key]["QUESTION"]]
        answer = [token for token in jokes_dict[key]["ANSWER"]]
        processed_corpus.append(question + answer)
    
    # return list of lists (every inner list is one joke)
    return processed_corpus


# Makes and returns a corpus where all the words have an id
def word_to_id(processed_corpus):
    dictionary = gensim.corpora.Dictionary(processed_corpus)
    return dictionary


# Converts the original word corpus with ids into vector corpus
def convert_corpus_vectors(processed_corpus, dictionary):
    vector_corpus = [dictionary.doc2bow(joke) for text in processed_corpus]
    return vector_corpus


# Makes model out of vector representations
def vector_model(vector_corpus, dictionary):
    model = models.TfidfModel(vector_corpus)
    return model


def what_jokes(text):
    pass


def why_jokes(text):
    pass


def generate_jokes(word, wh=None):
    accepted_wh = ["what", "why"]
    functions = [what_jokes, why_jokes]
    if wh is None:
        random.choice(functions)(word)
        print("Function chosen randomly.")
    else:
        for func in functions:
            func_name = str(func).split(" ")[1]
            func_wh = func_name.split("_")[0]
            if func_wh == wh:
                func(word)
                print("Function found.")
        if wh not in accepted_wh:
            print("Function invalid.")


if __name__ == "__main__":
    jokes = read_jokes("input.txt")
    generate_jokes("chicken", "why")
    generate_jokes("chicken")
    generate_jokes("chicken", "hwölp")

    processed_corpus = process_corpus(jokes)
    dictionary = word_to_id(processed_corpus)

    all_features = []
    for joke in processed_corpus:
        tokenized_sent = nlp(" ".join(joke))
        for i, token in enumerate(tokenized_sent):
            token_features = get_features(token, i, tokenized_sent)
            all_features.append(token_features)

    vectorized_features = vectorize_features(all_features)
    vector_corpus = vectors_to_corpus(vectorized_features)

    model = vector_model(vector_corpus, dictionary)
    model_corpus = model[vector_corpus]
    
    #model = models.Word2Vec(processed_corpus, min_count=1, size=100, window=5)
    #model.save("word2vec.model")
