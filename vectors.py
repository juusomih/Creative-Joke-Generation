import gensim,pprint
from collections import defaultdict
from gensim import models



def getfile_as_list(file):
    jokes = []
    with open(file,"r") as f:
        for line in f:
            jokes.append(line)
    return jokes


# Lowercase each document, split it by white space
def lowercase_file():
    texts = [[word for word in document.lower().split()] for document in getfile_as_list("input.txt")]
    return texts

# Count word frequencies and #only keep words that appear more than once uses returns the a corpus ready for use
def count_freq():
    texts = getfile_as_list("input.txt")
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    
    processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
    return processed_corpus

#Makes and returns a corpus where all the words have an id
def word_to_id(processed_corpus):
    dictionary = gensim.corpora.Dictionary(processed_corpus)
    return dictionary

#makes a vector represantation for input
def vector_for_new(new_string):
    new_vec = word_to_id(count_freq()).doc2bow(new_string.lower().split())
    return new_vec

#converts the original word corpus with ids into vector corpus
def convert_corpus_vectors(processed_corpus,dictionary):
    vector_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
    return vector_corpus ##the output is a a list of tuples where first entry is the ID of the token and second is the freq count of the token 

##Don't know what this does currently was supposed to make model out of the  vectors but is acting weird with the output
def vector_model(vector_corpus,dictionary,texts):
    
    model = models.TfidfModel(vector_corpus)
    for text in texts:
        print(model[dictionary.doc2bow(text)])


texts = lowercase_file()
processed_corpus = count_freq()
dictionary = word_to_id(processed_corpus)
vector_corpus = convert_corpus_vectors(processed_corpus,dictionary)
vector_model(vector_corpus,dictionary,texts)