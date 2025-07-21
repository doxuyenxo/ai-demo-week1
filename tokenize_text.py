import re

def tokenize_text(text):
    text = text.lower()
    tokens = re.sub(r"[^\w\s]", "", text)
    return tokens.split()