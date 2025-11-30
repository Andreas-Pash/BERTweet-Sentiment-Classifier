import os
import sys
import warnings

import re
import html
import unicodedata
from typing import List, Counter
import logging
import emoji
from textblob import TextBlob

import nltk
from nltk.tokenize import TweetTokenizer, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag 
 

warnings.filterwarnings("ignore", category=FutureWarning)

# NLTK data downloads
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

def _load_lexicon(path: str):
    words = set()
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            # In Bing Liu's files, comment lines start with ';'
            if not line or line.startswith(";"):
                continue
            words.add(line.lower())
    return words


#############################
# DICTIONARIES
#############################
LEXICON_DIR = os.path.join(PROJECT_ROOT, "data", "opinion_lexicon")
POS_LEXICON = _load_lexicon(os.path.join(LEXICON_DIR, "positive-words.txt"))
NEG_LEXICON = _load_lexicon(os.path.join(LEXICON_DIR, "negative-words.txt"))


CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "i'm": "i am",
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "what's": "what is",
    "there's": "there is",
    "let's": "let us",
    "you're": "you are",
    "they're": "they are",
    "we're": "we are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
}

SLANG = {
    "lol": "laughing",
    "lmao": "laughing",
    "rofl": "laughing",
    "brb": "be right back",
    "idk": "i do not know",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "btw": "by the way",
    "thx": "thanks",
    "u": "you",
    "ur": "your",
    "gr8": "great",
    "b4": "before",
    "pls": "please",
}

STOPWORDS = set(stopwords.words("english"))
NEGATION_WORDS = { 'barely', 'can’t', 'couldn’t', 'didn’t', 'doesn’t', 'don’t',
            'hardly', 'isn’t', 'mustn’t', 'neither', 'never', 'no', 'nobody',
            'none', 'not', 'nothing', 'nowhere', 'scarcely', 'shouldn’t',
            'wasn’t', 'weren’t', 'won’t', 'wouldn’t' }
CONTRASTIVE_CONNECTIVES = {
    "but", "however", "though", "although", "yet", "still", "whereas", "nonetheless", "nevertheless"
}
SENTENCE_BOUNDARIES = {".", "!", "?"}





tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=False, reduce_len=True)
lemmatizer = WordNetLemmatizer()


def normalize_unicode(text: str) -> str:
    """Normalize weird unicode forms (smart quotes, etc.)."""
    return unicodedata.normalize("NFKC", text)


def remove_urls(text: str) -> str:
    """Remove or replace URLs."""
    return re.sub(r"http\S+|www\.\S+", " ", text)


def remove_mentions(text: str) -> str:
    """Remove @user mentions."""
    return re.sub(r"@\w+", " ", text)


def split_hashtag_to_words(hashtag: str) -> str:
    """Split simple CamelCase/snake_case hashtags into words."""
    tag = hashtag.lstrip("#")
    words = re.sub("([a-z])([A-Z])", r"\1 \2", tag)  # CamelCase
    words = words.replace("_", " ")                  # snake_case
    return words.lower()



def process_hashtags(text: str, mode: str = "split") -> str:
    """
    mode = "split"  -> replace #HashTag with 'hash tag'
    mode = "remove" -> delete hashtags
    mode = "keep"   -> leave as-is
    """
    if mode == "keep":
        return text

    if mode == "remove":
        return re.sub(r"#\w+", " ", text)

    # default: split
    def repl(match):
        return " " + split_hashtag_to_words(match.group(0)) + " "
    return re.sub(r"#\w+", repl, text)



def emoji_to_text(text: str) -> str:
    """Convert emojis to text descriptions."""
    text = emoji.demojize(text, language="en")
    # :smiling_face: -> smiling face
    text = text.replace(":", " ")
    return text


def remove_html_entities(text: str) -> str:
    """Convert HTML entities like &amp; and then clean leftovers."""
    text = html.unescape(text)
    # optional: remove angle brackets etc.
    return text


def to_lowercase(text: str) -> str:
    """Lowercase."""
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Remove punctuation but keep letters, digits, and spaces."""
    return re.sub(r"[^\w\s]", "", text)
    # \w = letters, digits, underscore
    # \s = whitespace
    # [^\w\s] = anything that is not a letter, digit, underscore, or space

def remove_digits(text: str) -> str:
    """Remove digits but keep letters, punctuation, and spaces."""
    return re.sub(r"\d", "", text)
    # \d = any digit (0-9)



#############################
# 2. NORMALIZATION FUNCTIONS
#############################

def expand_contractions(text: str, mapping=CONTRACTIONS) -> str:
    """Expand contractions: can't -> cannot."""
    pattern = re.compile("|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)))

    def replacer(m):
        return mapping.get(m.group(0), m.group(0))

    return pattern.sub(replacer, text)


def replace_slang(text: str, mapping=SLANG) -> str:
    """Replace slang/short forms: lol -> laughing."""
    tokens = text.split()
    return " ".join(mapping.get(t, t) for t in tokens)


def normalize_elongated_words(text: str) -> str:
    """Sooooo -> soo (reduce 3+ repeated chars to 2)."""
    def _norm(word):
        return re.sub(r"(.)\1{2,}", r"\1\1", word)

    return " ".join(_norm(w) for w in text.split())


def spelling_correction(text: str) -> str:
    """
    Use TextBlob to correct spelling.
    Careful: slow and can be over-aggressive.
    """
    return str(TextBlob(text).correct())

def handle_negations(text: str) -> str:
    """
    Join negation words with the next sentiment-bearing token.
    e.g. "not good" -> "not_good", "didn't like" -> "not_like".
    """
    tokens = word_tokenize(text.lower())
    new_tokens = []
    skip_next = False

    for i, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue

        # If token is a negation and followed by another token, combine them
        if token in NEGATION_WORDS and i + 1 < len(tokens):
            next_token = tokens[i + 1]
            # Skip punctuation or stopwords if necessary
            if re.match(r"^[a-zA-Z]+$", next_token):
                new_tokens.append(f"not_{next_token}")
                skip_next = True
                continue

        new_tokens.append(token)

    return " ".join(new_tokens)

#############################
# 3. TOKENIZATION
#############################

def tokenize_tweet(text: str) -> List[str]:
    """Tweet-aware tokenization."""
    return tweet_tokenizer.tokenize(text)


#############################
# 4. LEMMATIZATION & STOPWORDS
#############################

def _get_wordnet_pos(tag: str) -> str:
    """Map POS tag to wordnet POS."""
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Lemmatize a list of tokens with POS tags."""
    tagged = pos_tag(tokens)
    lemmas = []
    for word, pos in tagged:
        wn_pos = _get_wordnet_pos(pos)
        lemma = lemmatizer.lemmatize(word, pos=wn_pos)
        lemmas.append(lemma)
    return lemmas

def stem_tokens(tokens: list[str]) -> list[str]:
    """Apply stemming to a list of tokens using NLTK's PorterStemmer."""
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def remove_stopwords_tokens(tokens: List[str], keep_negations: bool = True) -> List[str]:
    """
    Remove stopwords; keep negations like 'not' if keep_negations=True.
    """
    filtered = []
    for t in tokens:
        if t in STOPWORDS and (not (keep_negations and t in NEGATION_WORDS)):
            continue
        filtered.append(t)
    return filtered
 
 
 
  
def sentence_lengths(text: str) -> tuple[int, int]:
    """
    Compute the min and max sentence length (in tokens) for a tweet.
    Sentences are split on `.`, `!`, or `?`.
    If there are no tokens, returns (0, 0).
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    sentences = re.split(r"[.!?]+", text)
    lengths = []

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        tokens = tokenize_tweet(sent)
        if tokens:
            lengths.append(len(tokens))

    if not lengths:
        return (0, 0)

    return min(lengths), max(lengths), sum(lengths) / len(lengths)


def handle_contrastive_connectives(tokens: list[str]) -> list[str]:
    """
    Mark tokens that appear in the scope of contrastive connectives
    like 'but', 'however', 'though' by prefixing them with 'BUT_'.

    Example:
        ["the", "movie", "was", "boring", "but", "the", "ending", "was", "great"]
        -> ["the", "movie", "was", "boring", "but",
            "BUT_the", "BUT_ending", "BUT_was", "BUT_great"]
    """
    marked = []
    in_contrast_scope = False

    for tok in tokens:
        # Start contrastive region
        if tok in CONTRASTIVE_CONNECTIVES:
            marked.append(tok)       # keep the connective itself
            in_contrast_scope = True
            continue

        # End contrastive region at sentence boundary
        if tok in SENTENCE_BOUNDARIES:
            marked.append(tok)
            in_contrast_scope = False
            continue

        # Inside contrastive region: mark token
        if in_contrast_scope:
            # avoid marking pure punctuation again, just in case
            if tok.strip():
                marked.append(f"BUT_{tok}")
            else:
                marked.append(tok)
        else:
            marked.append(tok)

    return marked

def handle_contrastive_connectives(text: str) -> str:
    """
    Marks tokens that appear *after* contrastive connectives like 'but', 'however', 'though'.
    Example:
        "The movie was boring but the ending was great"
        -> "The movie was boring but contrast_the contrast_ending contrast_was contrast_great"
    """
    tokens = word_tokenize(text.lower())
    new_tokens = []
    contrast_mode = False

    for token in tokens:
        # Toggle into contrastive mode when a connective appears
        if token in CONTRASTIVE_CONNECTIVES:
            new_tokens.append(token)
            contrast_mode = True
            continue

        if contrast_mode:
            # Prefix subsequent tokens until punctuation or sentence end
            if re.match(r"[.!?]", token):
                contrast_mode = False
                new_tokens.append(token)
            else:
                new_tokens.append(f"contrast_{token}")
        else:
            new_tokens.append(token)

    return " ".join(new_tokens)


#############################
# 10. FULL PIPELINE HELPERS
############################# 

def light_clean(text: str) -> str:
    """Minimal normalisation for transformer inputs.
    Applies Unicode normalisation, removes URLs, and converts emojis to text. 
    (only a subset compared to the previous approach because bert takes care of mosts tuff)
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = normalize_unicode(text)
    text = remove_urls(text)
    text = emoji_to_text(text)
    
    return text

def pre_process(
    text: str,
    hashtag_mode: str = "split",
    remove_punct_flag: bool = False,
    add_lexicon_tokens: bool = True,
    add_bigrams: bool = True,
    tweet_text_len: bool = False
    # add_trigrams: bool = True,
) -> List[str]:
    """
    High-level helper:
    applies most of the above steps and returns final tokens. (selection was based on trial and error)
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    
    if tweet_text_len:
        min_len, max_len, mean_len = sentence_lengths(text)
        min_length_token = f"MIN_LEN<{int(min_len/5)}>"
        max_length_token = f"MAX_LEN<{int(max_len/5)}>"
        mean_length_token = f"MEAN_LEN<{int(mean_len/5)}>"
    
    # Cleaning
    text = normalize_unicode(text) 
    text = remove_urls(text)
    # text = remove_mentions(text)
    text = process_hashtags(text, mode=hashtag_mode)
    text = emoji_to_text(text)
    text = to_lowercase(text)
    text = remove_digits(text)
    # text = handle_negations(text)
    # text = handle_contrastive_connectives(text)
    
    if remove_punct_flag:
        text = remove_punctuation(text)


    # Normalization
    # text = expand_contractions(text)
    text = replace_slang(text)
    # text = normalize_elongated_words(text) 

    # Tokenization + lemmatization + stopwords
    tokens = tokenize_tweet(text)
    tokens = lemmatize_tokens(tokens)
    # tokens = remove_stopwords_tokens(tokens, keep_negations=True)

    if add_bigrams:
        bigrams_tokens = [ f'{tokens[i]} {tokens[i+1]}' 
            for i, txt in enumerate(tokens)
            if i<= len(tokens) - 2
            ] 
        tokens += bigrams_tokens

    # if add_trigrams:
    #     trigrams_tokens = [ str((tokens[i], tokens[i+1], tokens[i+2])) 
    #         for i, txt in enumerate(tokens)
    #         if i<= len(tokens) - 3
    #         ] 
    #     tokens += trigrams_tokens


    if add_lexicon_tokens:
        pos_hits = sum(1 for t in tokens if t in POS_LEXICON)
        neg_hits = sum(1 for t in tokens if t in NEG_LEXICON)

        # simplest version: repeat labels according to counts
        tokens += ["LEX_POS"] * pos_hits
        tokens += ["LEX_NEG"] * neg_hits 
        
    if tweet_text_len and min_len is not None and max_len is not None:
        tokens += [min_length_token, max_length_token, mean_length_token]
    
    return tokens



def to_feature_vector(tweet_tokens):
    # Should return a dictionary containing features as keys, and weights as values
    '''
    This function takes a tweet's tokens (preprocessed tweet text using
    the `pre_process` function ) and then counts how many times each token
    appears. It returns a dictionary where each key is a token (feature)
    and each value is its frequency (weight) within the tweet.

    For example:
        Input:  "Love this concert! Love it!"
        Output: {'love': 2, 'this': 1, 'concert': 1, 'it': 1}

    Returns
    -------
    dict
        A dictionary mapping tokens (features) to their counts (weights).
    '''
    return dict(Counter(tweet_tokens))


def train_test_split_data(raw_data, percentage):
    """Split the data between train_data and test_data according to the percentage
    and performs the preprocessing."""
    num_samples = len(raw_data)
    num_training_samples = int((percentage * num_samples))
    train_data = []
    test_data = []

    for (text, label) in raw_data[:num_training_samples]:
        train_data.append( (text, label) )
    for (text, label) in raw_data[num_training_samples:]:
        test_data.append( (text, label) )
    
    return train_data, test_data
 
 
