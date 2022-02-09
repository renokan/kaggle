"""
Utilities for toxic comment competitions:
* Jigsaw Rate Severity of Toxic Comments (2021-2022)
  https://www.kaggle.com/c/jigsaw-toxic-severity-rating
* Toxic Comment Classification Challenge (2018)
  https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
"""

import os
import numpy as np
import pandas as pd

import calendar
import textwrap
import re

from string import punctuation
from bs4 import BeautifulSoup

from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def get_samples(data: pd.DataFrame, frac_n: "float or int") -> pd.DataFrame:
    """ Get the selected piece of data. """
    result = data.copy()
    rs = 12345

    if frac_n > 0 and frac_n < 1:
        result = result.sample(frac=frac_n, random_state=rs)
    elif frac_n > 1 and frac_n < 100:
        frac_n = frac_n / 100
        result = result.sample(frac=frac_n, random_state=rs)
    elif frac_n >= 100:
        result = result.sample(n=frac_n, random_state=rs)
    else:
        # 0 or 1
        raise ValueError("Invalid '{}' value!".format(frac_n))

    return result.sort_index()


def get_stopwords(english: bool = True) -> list:
    """ Get the list of stop words. """
    result = ['wikipedia', 'wiki', 'utc']
    result = result + [w.lower() for w in calendar.month_name[1:]] \
                    + [w.lower() for w in calendar.month_abbr[1:]]

    if english:
        result = result + stopwords.words('english') \
                        + ["can't", "i'm"]
    
    return result


def get_words_and_tags(data: pd.Series, tagset: str = 'universal') -> pd.DataFrame:
    """ Extract information (words and tags) from data. """
    words_and_tags = []
    tag_col = "tag"
    word_col = "word"
    
    for string in data.values:
        words_and_tags.extend(pos_tag(string.split(),
                              tagset=tagset))
    
    result = pd.DataFrame.from_records(words_and_tags,
                                       columns=[word_col, tags_col])

    result[tags_col] = result[tags_col].astype("category")
    
    return result


def get_data_profile(data: pd.Series, round_rate: int = 3, scores: bool = False) -> pd.DataFrame:
    """ Extract information (stats and rate) from data. """
    result = []
    
    if scores:
        sid = SentimentIntensityAnalyzer()
    
    for index, value in data.items():
        text = re.sub(' +', ' ', value.replace('\n', ' '))
        text_len = len(text)

        tokens = text.split()
        n_tokens = len(tokens)

        rate_base = 0
        if n_tokens and text_len:
            rate_base = n_tokens / text_len
            rate_base = round(rate_base, round_rate)
                    
        rate_emotion = 0
        if tokens:
            n_isupper = sum([token.isupper() for token in tokens])
            if n_isupper:
                rate_emotion = n_isupper / n_tokens
                rate_emotion = round(rate_emotion, round_rate)

        rate_punctuation = 0
        if text:
            n_punctuations = len(re.findall(r"[?!:#$]", text))
            if n_punctuations:
                rate_punctuation = n_punctuations / text_len
                rate_punctuation = round(rate_punctuation, round_rate)

        value_info = [text_len, n_tokens, rate_base, rate_emotion, rate_punctuation]

        if scores:
            sid_scores = sid.polarity_scores(value)
            value_info.extend(sid_scores.values())
        
        result.append([index, value] + value_info)
        
    cols_list = ['index', 'text',
                 'len', 'tokens', 'base rate',
                 'upper rate', 'punct rate']

    if scores:
        cols_list.extend(sid_scores.keys())
    
    return pd.DataFrame(result, columns=cols_list).set_index('index')


def shorten_text(string: str, max_len: int) -> str:
    """ Shorting the text. """        
    string = textwrap.shorten(string,
                              width=max_len, placeholder='')   
    
    return string

        
def cleaning_text(string: str, stop_words: list = None) -> str:
    """ Cleaning the text. """
    text = string.lower().replace('\n', ' ').strip()

    text = re.sub(' +', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'image|file|jpg|jpeg', '', text)
    # Cut IP-address
    text = re.sub(r'\d{1,4}\.\d{1,4}\.\d{1,4}\.\d{1,4}', '', text)
    # Cut time, period or year
    text = re.sub(r'\d{2,}[:|-]\d{2,}|\d{4}', '', text)
    # Cut 20th or 1st
    text = re.sub(r'\d{1,}[th|st]', '', text)
    # Cut money
    text = re.sub(r'\d{1,}[,|\.]\d{2,}', '', text)
    # Cut address (9/169)
    text = re.sub(r'\d{1,}/\d{1,}', '', text)
    
    soup = BeautifulSoup(text, 'lxml')
    text = soup.get_text()
      
    words_cleaned = [w.strip(punctuation) for w in text.split() if not w.isdigit()]

    temp_list = []
    for word in words_cleaned:
        if word.isdigit():
            continue
        
        # "word!!!!!!word!!?!?!!"
        word_splitted = re.split('\?|!|:|;|\||\)|\(|\+|"|\.|,|#|&|_', word)
        
        if len(word_splitted) > 1:
            for w in word_splitted:
                w = w.strip(punctuation)
                if not w.isdigit():
                    temp_list.append(w)
        else:
            temp_list.append(word)

    # skip word "uhbsirtubgyihihlkjngkjbnkgjnbkf"
    max_word_len = 30
    words_cleaned = [w for w in temp_list if len(w) < max_word_len]

    # skip words with numbers
    words_cleaned = [w for w in words_cleaned if not bool(re.search(r'\d', w))]
    words_cleaned = [w for w in words_cleaned if bool(re.search(r"[a-zA-Z'\-]", w))]

    # skip one letter
    words_cleaned = [w for w in words_cleaned if len(w) > 1 or w in ['i', 'a']]

    text = " ".join(words_cleaned)
    
    if stop_words:
        text = " ".join([w for w in text.split()
                             if w not in stop_words])
        
    return text


def text_preprocessor(data: pd.Series, max_str_len: int = None,
                      stop_words: list = None, stemmer: bool = False) -> pd.Series:
    """ Preprocessing of text data. """
    result = data.copy().map(cleaning_text)
    
    if max_str_len:
        result = result.apply(shorten_text,
                              max_len=max_str_len)

    if stop_words:
        result = result.apply(lambda s: " ".join([w for w in s.split()
                                                  if w not in stop_words]))

    if stemmer:
        porter = PorterStemmer()
        
        result = result.apply(lambda s: " ".join([porter.stem(w) for w in s.split()]))
    
    return result
