"""
    LDA-T3113 Creative Natural Language Generation

    Final Project: Joke Generation

    Juuso-Miikka Heikkilä, Tiina Koho, Suvi Tapper

"""

import random
import spacy


def read_jokes(file):
    jokes = []
    with open(file, "r") as f:
        for line in f:
            jokes.append(line)
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
    jokes = read_jokes("input.txt")
    generate_jokes("chicken", "why")
    generate_jokes("chicken")
    generate_jokes("chicken", "hwölp")
