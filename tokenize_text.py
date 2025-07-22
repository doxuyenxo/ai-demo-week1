import re

def tokenize_text(text):
    text = text.lower()
    tokens = re.sub(r"[^\w\s]", "", text) #Remove special char
    return tokens.split()