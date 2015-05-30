#!/usr/bin/python2
#encoding: utf-8

#Nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

#Json libraries
import json
from pprint import pprint

#Setting stemmer
stemmer = SnowballStemmer("spanish")
stop = stopwords.words("spanish")

def freq(word, doc):
    return doc.count(word)

def word_count(doc):
    return len(doc)

def tf(word, doc):
    return (freq(word, doc) / float(word_count(doc)))

def num_docs_containing(word, list_of_docs):
    count = 0
    for document in list_of_docs:
        if freq(word, document) > 0:
            count += 1
    return 1 + count

def idf(word, list_of_docs):
    return math.log(len(list_of_docs) /
            float(num_docs_containing(word, list_of_docs)))

def tf_idf(word, doc, list_of_docs):
    return (tf(word, doc) * idf(word, list_of_docs))

def stem_json(jsonFile):
  data = json.load(open(jsonFile))
  cleanNew = ''
  for item in data:
    for word in item['noticia'].split():
      if word not in stop:
        stemming_word = stemmer.stem(word)
        cleanNew +=  stemming_word+' '
    item['noticia'] = cleanNew
  with open('json_stemming.json', 'w') as outfile:
    json.dump(data, outfile)

stem_json('example.json')