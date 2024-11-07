import spacy 
import numpy as np 
import pickle 
import nltk 

eng_nlp_sent = spacy.load("en_core_web_sm")
eng_nlp_sent.add_pipe("sentencizer")

def generate_test_set(
    gutenberg_file_name, num_samples, rng, min_words=100, max_words=100
):
    source_raw = nltk.corpus.gutenberg.raw(gutenberg_file_name)
    doc = eng_nlp_sent(source_raw)
    all_sents = [sent.text.replace("\n", " ") for sent in doc.sents]
    sent_lengths = [len(text.split()) for text in all_sents]
    compatible_spans = []
    from tqdm import tqdm 
    for i in tqdm(range(0,len(all_sents)),desc='Generating span lengths'):
        total_len = 0
        for j in range(i,len(all_sents)):
            total_len +=  sent_lengths[j] 
            if(total_len >= min_words):
                break 
        if(total_len >= min_words and total_len <= max_words):
            compatible_spans.append([i,j])
    print('found: %d compatible spans.'%len(compatible_spans))

    samples = []
    chosen_spans = rng.choice(len(compatible_spans), num_samples, replace=False)
    for span_index in chosen_spans: 
        i,j = compatible_spans[span_index]
        collated = " ".join( all_sents[i:j+1]  )
        span_words = len(collated.split())
        assert(span_words >= min_words and span_words <= max_words)
        samples.append(collated)
    return samples

def parse_args():
    import argparse 
    parser = argparse.ArgumentParser(description="Build a language corpus. Usage ex: python generate_dataset_fixed_length.py -i 150 -j 155 -n 50")
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
        '-n', '--num_spans',
        type=int,
        required=True,
        help="Number of spans to generate from each text."
    )
    return parser.parse_args()
args = parse_args()
MIN_WORDS = args.min_words
MAX_WORDS = args.max_words
NUM_SPANS = args.num_spans
rng = np.random.default_rng(3123)
sources = ["carroll-alice.txt","austen-sense.txt"]
output = {}
for source in sources:
    output[source] = generate_test_set(source, NUM_SPANS, rng,MIN_WORDS,MAX_WORDS)
fname = "test_sentences_"
if(MIN_WORDS==MAX_WORDS):
    fname+="%d_words.dat"%(MIN_WORDS)
else:
    fname+="%d_to_%d_words.dat"%(MIN_WORDS,MAX_WORDS)
with open(fname,'wb') as fh:
    pickle.dump(output,fh)