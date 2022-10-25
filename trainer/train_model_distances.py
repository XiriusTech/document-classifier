import csv
from operator import contains, mod
import re
import os
import glob
import pickle
import json
import unidecode
from annoy import AnnoyIndex


folder = "DATASET"

def perform_calculations(indices, wordsText, boundingPolies, cls):
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
            data["cls"] = cls
            list_distances.append(data)
            #print(w1["text"]+" <-> "+w2['text']+" "+str(distance_x)+" "+str(distance_y))
    return list_distances


docs = glob.glob(folder + '/**/*.json', recursive=True)
vocabulary = open("vocabulario.txt", "r", encoding="utf8")
voc_list = vocabulary.read().split('\n')
vocabulary_dict = {k: v for v, k in enumerate(voc_list)}

dataset = []
print("sarching vocabulary")
for doc in docs:
    path, filename = os.path.split(doc)
    classname = path.replace(
        folder+"\\", "").replace("\\", "_").replace(" ", "-")
    data = json.load(open(doc, encoding="utf8"))
    responses = data["pages"][:4]
    for r in responses:
        indices = []
        if "wordsText" in r.keys():
            wordsText = r['wordsText']
            boundingPolies = r['boundingPolies']
            for i in range(len(wordsText)):
                if unidecode.unidecode(wordsText[i].upper()) in voc_list:
                    indices.append(i)
            dataset.extend(perform_calculations(
                indices, wordsText, boundingPolies, classname))

f = 7
t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed
mapping = {}
for i, d in enumerate(dataset):
    mapping[i] = d['cls']
    v = [vocabulary_dict[d['w1']], vocabulary_dict[d['w2']],
         vocabulary_dict[d['w3']], d['dx2'], d['dy2'], d['dx3'], d['dy3']]
    t.add_item(i, v)
t.build(10)  # 10 trees
t.save('knn_model.ann')
with open("knn_model" + '.mapping', 'wb') as handle:
    pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("training model2")
