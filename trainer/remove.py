import glob
import os
import re

docs  = glob.glob("CARATULAS_DE_POLIZA_DOC_SEIZ" + '/**/*.json', recursive=True)
count=0
print(docs)
for doc in docs:
    r1 = re.match(".*to-([0-9]*).json",doc)
    if r1:
        if int(r1.groups()[0])<=5:
            count+=1
        else:
           os.remove(doc)
print(count)