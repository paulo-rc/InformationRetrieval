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

from collections import defaultdict

class MyCorpus(object):

  def __init__(self, json_path, json_key, dictionary_path='words_dict.dict', corpus_path='corpus.mm', corpus_tfidf_path='corpus_tfidf.mm'):
    self.json_path = json_path
    self.dictionary = []
    self.json_key = json_key
    self.corpus_tfidf_path = corpus_tfidf_path
    self.corpus_path = corpus_path
    self.dictionary_path = dictionary_path

  def __iter__(self):
    for line in self.load_json():
      yield self.dictionary.doc2bow(line[self.json_key].lower().split())

  def load_json(self):
    data = json.load(open(self.json_path))
    return data

  def remove_stop_words(self, documents_json):
    stoplist = stopwords.words("spanish")
    stoplist.extend(['no', '"no', '"desde', '"los', 'los', 'desde', 'pm', 'si', '|', '-'])
    stoplist = set('for a of the and to in'.split())
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

  def print_corpus(self, corpus_tfidf):
    for doc in corpus_tfidf:
      print doc

  def save_transform_corpus_to_tf_idf(self):
    corpus = self.load_vector_corpus()
    tfidf = models.TfidfModel(corpus)
    tfidf.save(self.corpus_tfidf_path)

  def load_transform_corpus_to_tf_idf(self, corpus):
    tfidf = models.TfidfModel.load(self.corpus_tfidf_path)
    corpus_tfidf = tfidf[corpus]
    return corpus_tfidf

  def first_run(self):
    data = self.load_json()
    data = self.remove_stop_words(data)
    data = self.remove_irrelevant_words(data)
    self.dictionary = corpora.Dictionary(data)
    self.save_dictionary()
    self.save_vector_corpus()
    self.save_transform_corpus_to_tf_idf()

  def query(self, query):
    vec_query = self.dictionary.doc2bow(query.lower().split())
    corpus_tfidf=corpus.load_transform_corpus_to_tf_idf(
      corpus=corpus.load_vector_corpus()
    )
    index = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=len(self.dictionary))
    tfidf = models.TfidfModel.load(self.corpus_tfidf_path)
    sims = index[tfidf[vec_query]]
    for calc in enumerate(sims[0:10]):
      print calc


corpus = MyCorpus(json_path='example.json', json_key='noticia')
corpus.load_dictionary()
#corpus.first_run()
#corpus.print_dictionary()
corpus.query('CNTPE tratara tema del sueldo')
# corpus_tfidf=corpus.load_transform_corpus_to_tf_idf(
#   corpus=corpus.load_vector_corpus()
# )
# corpus.print_corpus(
#   corpus_tfidf=corpus_tfidf
# )





#corpus.print_corpus(corpus_tfidf=corpus.load_vector_corpus())
#corpus.save_transform_corpus_to_tf_idf()
#corpus.print_corpus(corpus.load_transform_corpus_to_tf_idf(corp))
#corpus.load_dictionary('words_dict.dict')
#corpus.print_dictionary()
