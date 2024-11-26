#generates the entire PTB dataset, cleans it of (*), and 

import nltk

nltk.download('treebank')
nltk.download('brown')

def parse_args():
    import argparse 
    parser = argparse.ArgumentParser(description="Build a language corpus from the Penn Treebank subset availble in the NLTK. Usage ex: python generate_dataset_ptb_all.py -i 175 -j 150")
    # Argument for the number of simulations (integer)
    parser.add_argument(
        '-i', '--min_words',
        type=int,
        required=True,
        help="Minimum number of words in each span, inclusive."
    )
    parser.add_argument(
        '-j', '--max_words',
        type=int,
        required=True,
        help="Maximum number of words in each span, inclusive."
    )
    parser.add_argument(
        '-o', '--output_file',
        type=str,
        required=False,
        default='ptb_corpus.pkl',
        help="Output file name."
    )
    return parser.parse_args()
args = parse_args()
MIN_WORDS = args.min_words
MAX_WORDS = args.max_words
FNAME=args.output_file


from nltk.corpus import treebank_raw
raw = treebank_raw.raw()
articles = raw.split("START")
articles = [article.replace("\n"," ").replace('``', '"').replace("''", '"').replace('`', "'").replace("  "," ").replace("  "," ").replace("  "," ") for article in articles]
word_counts = [len(article.split(" ")) for article in articles]
min_wc,max_wc = 75,150
articles_filt = [article for article,wc in zip(articles,word_counts) if wc >= min_wc and wc <= max_wc]

import pickle 
with open(FNAME,'wb') as f:
    pickle.dump(articles_filt,f)