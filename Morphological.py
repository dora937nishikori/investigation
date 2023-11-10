import MeCab
import csv

def Morphological(file):
    with open(file,'r',encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        lines = []
        for text in csv_reader:
            #tagger = MeCab.Tagger(r'-Owakati -d "C:\mecab-ipadic-neologd"')
            #result = tagger.parse(text[0])
            m = MeCab.Tagger(r'-d "C:\mecab-ipadic-neologd"')
            parses = m.parse(text[0]).split('\n') #改行区切り
            morphs = [] #形態素解析
            removed = [] #空白連結

            for parse in parses:
                #空白区切り
                tmp = parse.split('\t')
                #EOS、空白行削除
                if len(tmp) > 1:
                    morphs.append(parse.split('\t'))

            #品詞抽出
            for morph in morphs:
                if '名詞' in morph[1]: 
                    removed.append(morph[0])
            

            lines.append(' '.join(removed))
    morphological_file = file[:-4]+'.txt'
    with open(morphological_file,'w' ,encoding='utf-8',newline='')as f:
        for row in lines:
            f.write(row+'\n')
    
    return morphological_file