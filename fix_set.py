import csv, spacy


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
        if tokenized[0].text.lower() in wh_words and all(w.text not in profanity for w in tokenized):
            with open("input.txt", "a") as output_file:
                output_file.write(tokenized.text + "\n")
