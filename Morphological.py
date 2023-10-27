import MeCab
import csv

def Morphological(file):
    with open(file,'r',encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        lines = []
        for text in csv_reader:
            tagger = MeCab.Tagger(r'-Owakati -d "C:\mecab-ipadic-neologd"')
            result = tagger.parse(text[0])

            lines.append(result)
    morphological_file = file[:-4]+'.txt'
    with open(morphological_file,'w' ,encoding='utf-8',newline='')as f:
        for row in lines:
            f.write(row)
    
    return morphological_file