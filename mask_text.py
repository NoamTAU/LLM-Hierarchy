# Command line installs:
# python -m spacy download en_core_web_sm


import spacy

eng_nlp = spacy.load("en_core_web_sm")
import numpy as np
import numpy.random

rng = np.random.default_rng(1337)


def mask_text(text, rng, probability,mask="XXXX",eng_nlp=None):
    if(eng_nlp is None):
        eng_nlp = spacy.load("en_core_web_sm")
    doc = eng_nlp(text)
    new_str = ""
    for token in doc:
        # print(token.text)
        if rng.uniform() < probability and token.text.isalpha():
            new_str += mask + token.whitespace_
        else:
            new_str += token.text_with_ws
    return new_str