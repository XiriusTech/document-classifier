from operator import contains, mod
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk import ngrams
from string import punctuation
import numpy as np
from nltk.stem import SnowballStemmer  
import os 
import glob
import fitz
import pickle
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import unidecode
from nltk.util import pr
from annoy import AnnoyIndex

folder = "CARATULAS_DE_POLIZA"

with open("knn_model"+".mapping","rb") as handle:
  mapping = pickle.load(handle)          
knn = AnnoyIndex(4, 'angular')
knn.load('knn_model.ann') # super fast, will just mmap the file
vocabulary = open("vocabulario.txt","r",encoding="utf8")
voc_list = vocabulary.read().split('\n')
vocabulary_dict = {k: v for v, k in enumerate(voc_list)}

def perform_calculations(list_words):
  if(len(list_words)<0):
    return []
  list_distances = []
  for i in range(1,len(list_words),1):
    w1 = list_words[i-1]
    w2 = list_words[i]
    height_w1 = w1['boundingPoly']['vertices'][-1]['y'] - w1['boundingPoly']['vertices'][0]['y']
    width_w1 = w1['boundingPoly']['vertices'][1]['x'] - w1['boundingPoly']['vertices'][0]['x']
    if height_w1>0 and width_w1>0:
      distance_x = (w2['boundingPoly']['vertices'][0]['x']-w1['boundingPoly']['vertices'][0]['x'])/width_w1
      distance_y = (w2['boundingPoly']['vertices'][0]['y']-w1['boundingPoly']['vertices'][0]['y'])/height_w1
      data = {}
      data["w1"]=unidecode.unidecode(w1['text'].upper())
      data["w2"]=unidecode.unidecode(w2['text'].upper())
      data["dx"]=distance_x
      data["dy"]=distance_y
      list_distances.append(data)
    #print(w1["text"]+" <-> "+w2['text']+" "+str(distance_x)+" "+str(distance_y))
  return list_distances

def classify_by_distances(distances,knn,mapping,vocabulary_dict):
  list_class = []
  for d in distances:
    v = [vocabulary_dict[d['w1']],vocabulary_dict[d['w2']],d['dx'],d['dy']]
    classification =  knn.get_nns_by_vector(v, 1, include_distances=True)
    list_class.append(mapping[classification[0][0]])
  if len(list_class)>0:
   print(list_class)
   return max(list_class, key = list_class.count)
  else:
    return "NO"

import csv
docs  = glob.glob(folder + '/**/*.json', recursive=True)
vocabulary = open("vocabulario.txt","r",encoding="utf8")
voc_list = vocabulary.read().split('\n')
print(voc_list)
dataset = []
print("sarching vocabulary")
for doc in docs:
 path,filename =  os.path.split(doc)
 classname = path.replace(folder+"\\","").replace("\\","_").replace(" ","-")
 data = json.load(open(doc, encoding="utf8"))
 responses = data["pages"][:4]
 for r in responses:
  contain_words = []
  if "text" in r.keys():
    words = r['words']
    for w in words:
      if  unidecode.unidecode(w['text'].upper()) in voc_list:
        contain_words.append(w)
    distances = perform_calculations(contain_words)
    class_ = classify_by_distances(distances,knn,mapping,vocabulary_dict) 