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

def perform_calculations(list_words,cls):
  if(len(list_words)<=0):
    return []
  list_distances = []
  for i in range(2,len(list_words),1):
    w1 = list_words[i-2]
    w2 = list_words[i-1]
    w3 = list_words[i]
    height_w1 = w1['boundingPoly']['vertices'][-1]['y'] - w1['boundingPoly']['vertices'][0]['y']
    width_w1 = w1['boundingPoly']['vertices'][1]['x'] - w1['boundingPoly']['vertices'][0]['x']
    if height_w1>0 and width_w1>0:
      distance_x2 = (w2['boundingPoly']['vertices'][0]['x']-w1['boundingPoly']['vertices'][0]['x'])/width_w1
      distance_y2 = (w2['boundingPoly']['vertices'][0]['y']-w1['boundingPoly']['vertices'][0]['y'])/height_w1
      distance_x3 = (w3['boundingPoly']['vertices'][0]['x']-w1['boundingPoly']['vertices'][0]['x'])/width_w1
      distance_y3 = (w3['boundingPoly']['vertices'][0]['y']-w1['boundingPoly']['vertices'][0]['y'])/height_w1
      data = {}
      data["w1"]=unidecode.unidecode(w1['text'].upper())
      data["w2"]=unidecode.unidecode(w2['text'].upper())
      data["w3"]=unidecode.unidecode(w3['text'].upper())
      data["dx2"]=distance_x2
      data["dy2"]=distance_y2
      data["dx3"]=distance_x3
      data["dy3"]=distance_y3
      data["cls"]=cls
      list_distances.append(data)
      #print(w1["text"]+" <-> "+w2['text']+" "+str(distance_x)+" "+str(distance_y))
  return list_distances

import csv
docs  = glob.glob(folder + '/**/*.json', recursive=True)
vocabulary = open("vocabulario.txt","r",encoding="utf8")
voc_list = vocabulary.read().split('\n')
vocabulary_dict = {k: v for v, k in enumerate(voc_list)}

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
    dataset.extend(perform_calculations(contain_words,classname))

f = 7
t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed
mapping={}
for i,d in enumerate(dataset):
    mapping[i]=d['cls']
    v = [vocabulary_dict[d['w1']],vocabulary_dict[d['w2']],vocabulary_dict[d['w3']],d['dx2'],d['dy2'],d['dx3'],d['dy3']]
    t.add_item(i, v)
t.build(10) # 10 trees
t.save('knn_finesa.ann')
with open("knn_finesa"+ '.mapping', 'wb') as handle:
   pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("training model2")