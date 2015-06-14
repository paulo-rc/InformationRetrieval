#!/usr/bin/python2
#encoding: utf-8

#Priority queue
import Queue as Q

#Nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

#Gensim
from gensim import corpora, models, similarities

#Json libraries
import json
from pprint import pprint

from collections import defaultdict

#Importing settings
from django.conf import settings

#Defining data dirs
json_dir = settings.GENSIM_DATA_JSON_DIR
data_dir = settings.GENSIM_GENERATED_DATA_DIR

class MyCorpus(object):

  def __init__(self, json_key, json_path=json_dir, dictionary_path=data_dir+'words_dict.dict',
    corpus_path=data_dir+'corpus.mm', corpus_tfidf_path=data_dir+'corpus_tfidf.mm',
    similarities_matrix_path=data_dir+'matrix'):
      self.json_path = json_path
      self.dictionary = []
      self.json_key = json_key
      self.corpus_tfidf_path = corpus_tfidf_path
      self.corpus_path = corpus_path
      self.dictionary_path = dictionary_path
      self.similarities_matrix_path = similarities_matrix_path

  def __iter__(self):
    for line in self.load_json():
      yield self.dictionary.doc2bow(line[self.json_key].lower().split())

  def load_json(self):
    data = json.load(open(self.json_path))
    return data

  def remove_stop_words(self, documents_json):
    stoplist = stopwords.words("spanish")
    stoplist.extend(['no', '"no', '"desde', '"los', 'los', 'desde', 'pm', 'si', '|', '-'])
    words_array = [[word for word in document['noticia'].lower().split() if word not in stoplist]
          for document in documents_json]
    return words_array

  def remove_irrelevant_words(self, words_array):
    frequency = defaultdict(int)
    for text in words_array:
      for token in text:
        frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1]
                    for text in words_array]
    return texts

  def save_vector_corpus(self):
    corpus = [text for text in self]
    corpora.MmCorpus.serialize(self.corpus_path, corpus)

  def load_vector_corpus(self):
    return corpora.MmCorpus(self.corpus_path)

  def save_dictionary(self):
      self.dictionary.save(self.dictionary_path)

  def load_dictionary(self):
    self.dictionary = corpora.Dictionary.load(self.dictionary_path)

  def print_dictionary(self):
    pprint(self.dictionary.token2id)

  def print_corpus(self):
    for doc in self.load_vector_corpus():
      print doc

  def print_corpus_tfidf(self):
    tfidf = self.load_transformed_corpus_to_tf_idf()
    for doc in tfidf[self.load_vector_corpus()]:
      print doc

  def save_transformed_corpus_to_tf_idf(self):
    corpus = self.load_vector_corpus()
    tfidf = models.TfidfModel(corpus)
    tfidf.save(self.corpus_tfidf_path)

  def load_transformed_corpus_to_tf_idf(self):
    return models.TfidfModel.load(self.corpus_tfidf_path)

  def load_similiarities_matrix(self):
    return similarities.Similarity.load(self.similarities_matrix_path)

  def save_similarities_matrix(self):
    corpus = self.load_vector_corpus()
    tfidf = self.load_transformed_corpus_to_tf_idf()
    index = similarities.Similarity(self.similarities_matrix_path, tfidf[corpus], num_features=len(self.dictionary))
    index.save(self.similarities_matrix_path)

  def first_run(self):
    data = self.load_json()
    data = self.remove_stop_words(data)
    #data = self.remove_irrelevant_words(data)
    self.dictionary = corpora.Dictionary(data)
    self.save_dictionary()
    self.save_vector_corpus()
    self.save_transformed_corpus_to_tf_idf()
    self.save_similarities_matrix()

  def query(self, query, begin, end):
    queue  = Q.PriorityQueue()
    vec_query = self.dictionary.doc2bow(query.lower().split())
    index = self.load_similiarities_matrix()
    for similarities in enumerate(index[vec_query][begin:end]) :
      if similarities[1] != 0:
        queue.put((-similarities[1], similarities[0]))
    while not queue.empty():
      print queue.get()
