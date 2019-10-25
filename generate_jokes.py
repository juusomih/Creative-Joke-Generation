import gensim
import pprint
import random
import spacy
import string

from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
from gensim import models


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
                jokes[keyword] = {"QUESTION":sents[0],          # question as string
                                  "ANSWER":sents[1],            # answer as string
                                  "QUESTION_LEMMAS":que_lemmas, # question as list of lemmas
                                  "ANSWER_LEMMAS":ans_lemmas}   # answer as list of lemmas
    return jokes


# Makes a corpus that is used to build the id dictionary.
def process_corpus(jokes_dict):
    processed_corpus = []

    # Combine question lemma lists and answer lemma lists
    for key in jokes_dict:
        question = [token for token in jokes_dict[key]["QUESTION_LEMMAS"]]
        answer = [token for token in jokes_dict[key]["ANSWER_LEMMAS"]]
        processed_corpus.append(question + answer)
    
    # return list of lists (every inner list is one joke)
    return processed_corpus


# Makes and returns a corpus where all the words have an id
def word_to_id(processed_corpus):
    dictionary = gensim.corpora.Dictionary(processed_corpus)
    return dictionary


# Converts the original word corpus with ids into vector corpus
def convert_corpus_vectors(processed_corpus, dictionary):
    vector_corpus = [dictionary.doc2bow(joke) for joke in processed_corpus]
    return vector_corpus


# Makes model out of vector representations
def vector_model(vector_corpus, dictionary, processed_corpus):
    pass
    model = models.TfidfModel(vector_corpus)
    for joke in processed_corpus:
        print("---\n", joke)
        pprint.pprint(model[dictionary.doc2bow(joke)])


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
    jokes = read_jokes("testiset.txt")
    generate_jokes("chicken", "why")
    generate_jokes("chicken")
    generate_jokes("chicken", "hwölp")

    processed_corpus = process_corpus(jokes)
    dictionary = word_to_id(processed_corpus)
    vector_corpus = convert_corpus_vectors(processed_corpus, dictionary)
    vector_model(vector_corpus, dictionary, processed_corpus)
