#!/usr/bin/python2
#encoding: utf-8

#Nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

#Gensim
from gensim import corpora, models, similarities

#Json libraries
import json
from pprint import pprint

#utility for remove words that appear only once
from collections import defaultdict

def load_json(path):
  data = json.load(open(path))
  return data

def remove_stop_words(documents_json):
  stoplist = stopwords.words("spanish")
  words_array = [[word for word in document['noticia'].lower().split() if word not in stoplist]
              for document in documents_json]
  return words_array

def remove_irrelevant_words(words_array):
  frequency = defaultdict(int)
  for text in words_array:
    for token in text:
      frequency[token] += 1
  texts = [[token for token in text if frequency[token] > 1]
         for text in words_array]
  return texts

def wrapper():
  data = load_json('example.json')
  data = remove_stop_words(data)
  #data = remove_irrelevant_words(data)
  dictionary = corpora.Dictionary(data)
  pprint(dictionary.token2id)
  #pprint(data)

wrapper()



