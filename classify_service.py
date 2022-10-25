import re
import pickle
import config
import unidecode

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
  
  def classify(self, ocr_response):
    dict_resp = {}
    responses = ocr_response["pages"][:4]
    list_class = []

    for r in responses:
      indices = []
      distances = []
      wordsText = r['wordsText']
      boundingPolies = r['boundingPolies']
      for i in range(len(wordsText)):
          if unidecode.unidecode(wordsText[i].upper()) in self.voc_list:
              indices.append(i)
      distances.extend(self.perform_calculations(
          indices, wordsText, boundingPolies))
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
    
  def perform_calculations(self,indices, wordsText, boundingPolies):
    if (len(indices) <= 0):
        return []
    list_distances = []
    for i in range(2, len(indices), 1):
        indice1 = indices[i-2]
        indice2 = indices[i-1]
        indice3 = indices[i]
        height_w1 = boundingPolies[indice1][-1]['y'] - \
            boundingPolies[indice1][0]['y']
        width_w1 = boundingPolies[indice1][1]['x'] - \
            boundingPolies[indice1][0]['x']
        if height_w1 > 0 and width_w1 > 0:
            distance_x2 = (
                boundingPolies[indice2][0]['x']-boundingPolies[indice1][0]['x'])/width_w1
            distance_y2 = (
                boundingPolies[indice2][0]['y']-boundingPolies[indice1][0]['y'])/height_w1
            distance_x3 = (
                boundingPolies[indice3][0]['x']-boundingPolies[indice1][0]['x'])/width_w1
            distance_y3 = (
                boundingPolies[indice3][0]['y']-boundingPolies[indice1][0]['y'])/height_w1
            data = {}
            data["w1"] = unidecode.unidecode(wordsText[indice1].upper())
            data["w2"] = unidecode.unidecode(wordsText[indice2].upper())
            data["w3"] = unidecode.unidecode(wordsText[indice3].upper())
            data["dx2"] = distance_x2
            data["dy2"] = distance_y2
            data["dx3"] = distance_x3
            data["dy3"] = distance_y3
            list_distances.append(data)
            #print(w1["text"]+" <-> "+w2['text']+" "+str(distance_x)+" "+str(distance_y))
    return list_distances