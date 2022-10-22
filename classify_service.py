import re
from nltk import classify
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import config
from sklearn.feature_extraction.text import TfidfVectorizer
import unidecode
import sentences_processor as sent_processor
from annoy import AnnoyIndex

class ClasifyService:

  def __init__(self):
    print("Loadind models for classification process")
    with open(config.MODEL_FOLDER+"/"+config.MAPPING_DICT,"rb") as handle:
      self.mapping = pickle.load(handle) 
    self.knn = AnnoyIndex(7, 'angular')
    self.knn.load(config.MODEL_FOLDER+"/"+config.ANNOY_FILENAME) # super fast, will just mmap the file
    vocabulary = open(config.VOCAB_FILENAME,"r",encoding="utf8")
    self.voc_list = vocabulary.read().split('\n')
    self.vocabulary_dict = {k: v for v, k in enumerate(self.voc_list)}

  def most_frequent(self,List):
    return max(set(List), key = List.count)
  
  def classify(self, seiz_response):
    dict_resp = {}
    responses = seiz_response["pages"][:4]
    list_class = []
    for r in responses:
      list_words  =  self.get_list_wors_from_response(r)
      distances = self.perform_distaces(list_words)
      for d in distances:
        v = [self.vocabulary_dict[d['w1']],self.vocabulary_dict[d['w2']],self.vocabulary_dict[d['w3']],d['dx2'],d['dy2'],d['dx3'],d['dy3']]
        classification =  self.knn.get_nns_by_vector(v, 1, include_distances=True)
        if classification[1][0]<2:
          print(str(d)+" CLASE "+self.mapping[classification[0][0]])
          list_class.append(self.mapping[classification[0][0]])
    if len(list_class)>0:
       print(list_class)
       dict_resp["class"] = max(list_class, key = list_class.count)
       print(dict_resp["class"])
    else:
       dict_resp["class"] = "NOT_FOUND"
    return dict_resp

  def get_list_wors_from_response(self, response):
    list_words_in_vocabulary = []
    words = response['words']
    for w in words:
      if  unidecode.unidecode(w['text'].upper()) in self.voc_list:
        list_words_in_vocabulary.append(w)
    return list_words_in_vocabulary
    
  def perform_distaces(self,list_words):
    if(len(list_words)<0):
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
        list_distances.append(data)
      #print(w1["text"]+" <-> "+w2['text']+" "+str(distance_x)+" "+str(distance_y))
    return list_distances