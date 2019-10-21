import nltk,spacy,csv
from nltk.tokenize import word_tokenize

jokes = []
onejoke = []
nlp = spacy.load("en")

#testi = nlp("Why did the pasta chef take his car into the body shop? Cause it got al dente'd up!")
#for token in testi:
#    print(token)

with open("shortjokes.csv","r") as file:
    
    csv_reader = csv.reader(file,delimiter=",")
    line_count = 0
    for row in csv_reader:
        tokenized = nlp(row[1])
        for token in tokenized:
            
            onejoke.append(token.text)

          
        if(onejoke[0]=="What" or onejoke[0]=="Who" or onejoke[0]=="Why" or onejoke[0]=="How" or onejoke[0]=="When"):
            
            with open("output.txt","a") as f:
                f.write(tokenized.text+"\n")
                
                

            
            
        
        onejoke = []


        
        
        
    
    
    
#print(len(jokes))