import random
import spacy
import string

from nltk.tokenize import sent_tokenize, word_tokenize


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
                que_lemmas = [token.lemma_ for token in que_tokens if not token.is_stop and token.is_alpha]
                ans_lemmas = [token.lemma_ for token in ans_tokens if not token.is_stop and token.is_alpha]

                # Dictionary key is a noun lemma from question
                jokes[keyword] = {"QUESTION":sents[0],          # question as string
                                  "ANSWER":sents[1],            # answer as string
                                  "QUESTION_LEMMAS":que_lemmas, # question as list of lemmas
                                  "ANSWER_LEMMAS":ans_lemmas}   # answer as list of lemmas
    return jokes


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
    jokes = read_jokes("test_input.txt")
    print(jokes)
    generate_jokes("chicken", "why")
    generate_jokes("chicken")
    generate_jokes("chicken", "hwölp")
