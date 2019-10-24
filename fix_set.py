import csv, spacy

jokes = []
onejoke = []
nlp = spacy.load("en_core_web_sm")

#
def sortjokes_tofile(inputfile,outputfile): 
    with open(inputfile,"r") as file:
        
        csv_reader = csv.reader(file,delimiter=",")
        
        for row in csv_reader:
            tokenized = nlp(row[1])
            for token in tokenized:
                onejoke.append(token.text)
            if(onejoke[0]=="What" or onejoke[0]=="Who" or onejoke[0]=="Why" or onejoke[0]=="How" or onejoke[0]=="When"):
                
                with open("outputfile","a") as f:
                    f.write(tokenized.text+"\n")
            onejoke = []

#Reads jokes from a file splits them into question and answer parts by splitting on a "?" if no "?" skips the joke and saves them into a dictionary.
def jokes_into_question_answer(inputfile):
    parts = {}
    with open(inputfile,"r") as file:
        
        for line in file:
            
            osat = line.split("?")
            if(len(osat)== 2):
                parts[osat[0]] = osat[1]

    return parts

# Load spacy module
nlp = spacy.load("en")

# List of wh-words
wh_words = ["what", "who", "why", "how", "when"]

# Extract profanity words from file
profanity = []
with open("profanity_words.txt", "r") as f:
    for row in f:
        profanity.append(row.rstrip())

# Read original jokes file, write wanted jokes to new file
with open("shortjokes.csv", "r") as input_file:
    csv_reader = csv.reader(input_file, delimiter = ",")
    for row in csv_reader:
        tokenized = nlp(row[1])
        # Check that first word is wh-word and that none of the words is offensive
        if tokenized[0].lemma_ in wh_words and all(w.lemma_ not in profanity for w in tokenized):
            with open("input.txt", "a") as output_file:
                output_file.write(tokenized.text + "\n")
