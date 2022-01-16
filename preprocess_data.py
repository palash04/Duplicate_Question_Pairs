import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

STOPWORDS = set(stopwords.words('english'))

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

import string

PUNCT_TO_REMOVE = string.punctuation

from spacy.lang.en import English

nlp = English()
tokenizer = nlp.tokenizer


def tokenize(text):
    tokens = tokenizer(text)
    return tokens


class Vocabulary(object):
    def __init__(self, train_df):
        self.train_df = train_df
        self.vocab = set()
        self.vocab.add('<UNK>')
        self.vocab.add('<PAD>')
        self.stoi = {'<PAD>': 0, '<UNK>': 1}
        self.itos = {0: '<PAD>', 1: '<UNK>'}

    def build_vocabulary(self):
        train_df = self.train_df
        idx = len(self.vocab)
        for i in tqdm(range(len(train_df))):
            q1 = train_df.iloc[i, 3]
            q2 = train_df.iloc[i, 4]
            q1_tokens = tokenize(q1.lower())
            q2_tokens = tokenize(q2.lower())

            for token in q1_tokens:
                if str(token) not in self.stoi:
                    self.vocab.add(str(token))
                    self.stoi[str(token)] = idx
                    self.itos[idx] = str(token)
                    idx += 1

            for token in q2_tokens:
                if str(token) not in self.stoi:
                    self.vocab.add(str(token))
                    self.stoi[str(token)] = idx
                    self.itos[idx] = str(token)
                    idx += 1


def get_vocab_and_embed_matrix(train_df, embed_dim=100):
    vocab = Vocabulary(train_df)
    print('Building vocabulary')
    vocab.build_vocabulary()
    glove_path = f'glove.6B.{embed_dim}d.txt'
    g_vocab = {}
    with open(glove_path, 'r', encoding="utf-8") as f:
        data = f.read().strip().split('\n')
    for i in range(len(data)):
        word_i = data[i].split(' ')[0]
        embeddings_i = [float(val) for val in data[i].split(' ')[1:]]
        g_vocab[word_i] = embeddings_i

    pad_emb_np = np.zeros((1, embed_dim))
    unk_emb_np = np.random.normal(scale=0.6, size=(1, embed_dim))
    g_vocab['<PAD>'] = pad_emb_np
    g_vocab['<UNK>'] = unk_emb_np
    vocab_size = len(vocab.vocab)
    embed_matrix = torch.zeros((vocab_size, embed_dim))

    for k, v in vocab.stoi.items():
        if k in g_vocab:
            embed_matrix[v] = torch.tensor(np.array(g_vocab[k]))
        else:
            embed_k = np.random.normal(scale=0.6, size=(1, embed_dim))
            embed_matrix[v] = torch.tensor(embed_k)
    return vocab, embed_matrix


def handle_nans(df, drop):
    is_NaN = df.isnull()
    row_has_NaN = is_NaN.any(axis=1)
    rows_with_NaN = df[row_has_NaN]
    if len(rows_with_NaN) > 0:
        if drop:
            ids = rows_with_NaN.index.values
            df = df.drop(ids)
            return df
        else:
            ids = rows_with_NaN['test_id'].values.astype(int)
            for test_id in ids:
                if pd.isnull(rows_with_NaN.loc[test_id, 'question1']):
                    df.loc[test_id, 'question1'] = ''
                if pd.isnull(rows_with_NaN.loc[test_id, 'question2']):
                    df.loc[test_id, 'question2'] = ''
            return df
    else:
        return df


contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot",
                       "'cause": "because", "could've": "could have", "couldn't": "could not",
                       "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                       "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
                       "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                       "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                       "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would",
                       "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
                       "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
                       "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us",
                       "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                       "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                       "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                       "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                       "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                       "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                       "she'll've": "she will have", "she's": "she is", "should've": "should have",
                       "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                       "so's": "so as", "this's": "this is", "that'd": "that would", "that'd've": "that would have",
                       "that's": "that is", "there'd": "there would", "there'd've": "there would have",
                       "there's": "there is", "here's": "here is", "they'd": "they would",
                       "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
                       "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
                       "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                       "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
                       "what'll've": "what will have", "what're": "what are", "what's": "what is",
                       "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
                       "where's": "where is", "where've": "where have", "who'll": "who will",
                       "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is",
                       "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                       "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                       "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
                       "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
                       "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                       "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center',
                       'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling',
                       'theatre': 'theater', 'cancelled': 'canceled'}


def clean_contractions(text, mapping):
    text = text.lower()
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join(
        [mapping[t] if t in mapping else mapping[t.lower()] if t.lower() in mapping else t for t in text.split(" ")])
    return text


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


def remove_stopwords(text):
    return ' '.join([word for word in str(text).split() if word not in STOPWORDS])


"""
Stopwords:

i, me, my, myself, we, our, ours, ourselves, you,
you're, you've, you'll, you'd, your, yours, yourself,
yourselves, he, him, his, himself, she, she's, her, hers,
herself, it, it's, its, itself, they, them, their, theirs,
themselves, what, which, who, whom, this, that, that'll, these,
those, am, is, are, was, were, be, been, being, have, 
has, had, having, do, does, did, doing, a, an, the, and, but, 
if, or, because, as, until, while, of, at, by, for, with, about, 
against, between, into, through, during, before, after, above, below, to, 
from, up, down, in, out, on, off, over, under, again, further, then, once,
here, there, when, where, why, how, all, any, both, each, few, more, most, other, 
some, such, no, nor, not, only, own, same, so, than, too, very, s, t, can, will, 
just, don, don't, should, should've, now, d, ll, m, o, re, ve, y, ain, aren, aren't, 
couldn, couldn't, didn, didn't, doesn, doesn't, hadn, hadn't, hasn, hasn't, haven, 
haven't, isn, isn't, ma, mightn, mightn't, mustn, mustn't, needn, needn't, shan, 
shan't, shouldn, shouldn't, wasn, wasn't, weren, weren't, won, won't, wouldn, wouldn't"

"""


def lemmatize_words(text):
    wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
    pos_tagged_text = nltk.pos_tag(text.split())
    return ' '.join(
        [lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])


def preprocess(df, drop=True):
    # drop nan or fill nan in test dataset
    df = handle_nans(df, drop)

    # clean contractions
    df['question1'] = df['question1'].apply(lambda x: clean_contractions(str(x), contraction_mapping))
    df['question2'] = df['question2'].apply(lambda x: clean_contractions(str(x), contraction_mapping))

    # remove punctuations
    df['question1'] = df['question1'].apply(lambda x: remove_punctuation(x))
    df['question2'] = df['question2'].apply(lambda x: remove_punctuation(x))

    # remove stopwords (but in this project it is not needed to remove them)
    # df['question1'] = df['question1'].apply(lambda x: remove_stopwords(x))
    # df['question2'] = df['question2'].apply(lambda x: remove_stopwords(x))

    # lemmatization
    # df['question1'] = df['question1'].apply(lambda x: lemmatize_words(x))
    # df['question2'] = df['question2'].apply(lambda x: lemmatize_words(x))

    return df


def sample_df(df, sample=50000):
    df_zeros = df[df['is_duplicate'] == 0]
    df_ones = df[df['is_duplicate'] == 1]

    df_zeros = df_zeros.sample(frac=1).reset_index(drop=True)
    df_ones = df_ones.sample(frac=1).reset_index(drop=True)

    df = pd.concat([df_zeros[:sample], df_ones[:sample]])
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def get_data_frames(sample_rows_per_class=50000):
    train_csv_path = 'train.csv'
    test_csv_path = 'test.csv'
    sample_submission_csv_path = 'sample_submission.csv'
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    sample_sub_df = pd.read_csv(sample_submission_csv_path)

    train_df = sample_df(train_df, sample=sample_rows_per_class)

    print('Preprocessing text')
    train_df = preprocess(train_df, drop=True)
    test_df = preprocess(test_df, drop=False)

    return train_df, test_df, sample_sub_df
