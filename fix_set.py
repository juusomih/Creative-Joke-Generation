import nltk,spacy,csv
from nltk.tokenize import word_tokenize

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

    #for i,j in parts.items():
    #   print("key "+i +" value"+j)


def main():
    jokes_into_question_answer("testiset.txt")


main()