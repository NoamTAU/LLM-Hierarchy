def parse_args():
    import argparse 
    parser = argparse.ArgumentParser(description="Build a language corpus from the Wikitext2 dataset. Usage ex: python generate_dataset_wiki2.py -m 175 -n 150")
    # Argument for the number of simulations (integer)
    parser.add_argument(
        '-n', '--min_words',
        type=int,
        nargs='+',
        required=True,
        help="Minimum number of words in each span, inclusive. Accepts multiple values as an array e.g. '-n 50 60 70'."
    )
    parser.add_argument(
        '-m', '--max_words',
        type=int,
        nargs='+',
        required=True,
        help="Maximum number of words in each span, inclusive. Accepts multiple values as an array  e.g. '-m 50 60 70'.."
    )
    parser.add_argument(
        '-o', '--output_file',
        type=str,
        required=False,
        default='wiki_corpus.pkl',
        help="Output file name. If there are multiple values for min_words and max_words, the output file will be named as [FILENAME]_minwords_maxwords.pkl."
    )
    parser.add_argument(
        '--strip_headers',
        action='store_true',
        help="Flag to indicate whether to strip headers from the articles. Default is False."
    )
    return parser.parse_args()
args = parse_args()
MIN_WORDS = args.min_words
MAX_WORDS = args.max_words
STRIP_HEADERS = args.strip_headers
FNAME=args.output_file
if len(MIN_WORDS) != len(MAX_WORDS):
    raise ValueError("The number of minimum and maximum words must be the same.")

from datasets import load_dataset
print("Loading dataset...")
ds_raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
print("Dataset loaded.")

#identifying articles in the wikitext dataset
import re
pattern = r'^[^=]*=[^=]*=[^=]*\n$'
source = ds_raw['train']['text']
matches = [(i,string) for i,string in enumerate(source) if re.match(pattern,string)]
matches = [(i,string) for i,string in matches if i > 0 and i < len(source)-2 and source[i-1] == "" and source[i+1] == ""]
article_ranges = [(matches[i][0],matches[i+1][0]) for i in range(len(matches)-1)]

#generating article strings:
articles = [source[start:end] for start,end in article_ranges]
if(STRIP_HEADERS):
    def is_header(string):
        return ( string[:3] == " = " and string[-4:] == " = \n" ) 
    articles = [ [string for string in article if not is_header(string)] for article in articles]

#preparing to detokenize the articles.
import sacremoses 
moses_detokenizer = sacremoses.MosesDetokenizer(lang='en')
def normalize_text_moses(str):
    output = moses_detokenizer.detokenize(str.split(' '), return_str=True)
    output = output.replace(" @,@ ",",") # this was a custom comma introduced by the wikitext-2 dataset for numbers.
    return output
from tqdm import tqdm
articles = [normalize_text_moses(''.join(article)) for article in tqdm(articles,desc="Normalizing text")]

#okay, now let's output the corpus, which we generate by splitting the article into sentences, and including sentences until the min/max words are reached.
#if we exceed the max word count, we skip to the next article, else we output the article to the corpus. 
#could be made more performant, probably don't want to be doing so much string adding, but I imagine the tokenization is the most expensive part anyway.
import nltk
def generate_corpus(min_words,max_words,articles):
    corp = []
    for article in tqdm(articles,desc='Finding correct length articles'):
        sentences = nltk.tokenize.sent_tokenize(article)
        art = ""
        for sent in sentences:
            art += sent + " "
            wc = len(art.split(" "))
            if( wc >= min_words):
                if( wc  <=  max_words):
                    corp.append(art)
                break 
    return corp 

import pickle 
for min_words,max_words in tqdm(zip(MIN_WORDS,MAX_WORDS),desc='Generating corporae of different lengths',total=len(MIN_WORDS)):
    corp = generate_corpus(min_words,max_words,articles)
    if(len(MIN_WORDS) == 1):
        filename = FNAME
    else:
        filename = FNAME + "_{}_{}".format(min_words,max_words)+'.pkl'
    with open(filename,'wb') as f:
        pickle.dump(corp,f)