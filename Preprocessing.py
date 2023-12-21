import re
import csv
import emoji
import demoji
import neologdn
import unicodedata

#英文除去
def checkAlnum(word):
  alnum = re.compile(r'^[a-zA-Z0-9]+$')
  result = alnum.match(word) is not None
  return result

#前処理の中身
def preprocessing(text):
    #空白,改行文字など削除
    text = re.sub(r"\s", "", text)
    text = text.replace('\n','').replace('\r','')
    #アルファベット一文字のみを除去
    text = re.sub(r'(?<![a-z])[a-z](?![a-z])', '', text)
    #半角記号除去
    text = re.sub('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』﹃【】＆＊・（）＄＃＠。、？！｀＋￥％↓◆▽゚�❛ᴗ⌯¤̴̶̷̀ω¤̴̶̷́✧ฅ´∞∀♪ч☆ψ♡ヽ●ε●★›‹]','',text)
    #全角記号除去
    text = re.sub("[\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F]", '', text)
    #絵文字除去
    text = emoji.replace_emoji(text)
    text = demoji.replace(string=text, repl='')
    #文字種の統一
    text = unicodedata.normalize('NFKC',text)
    #小文字
    text = text.lower()
    #数字を0に置換
    text = re.sub(r'\d+', '0', text)
    #URL削除
    text = re.sub(r'http?://[\w/:%#\$&\?\(\)~\.=\+\-]+','', text)
    text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+','', text)
    #www除去
    text = re.sub(r'w+','',text)
    #連続長音記号除去
    text = neologdn.normalize(text)
    #...除去
    text = re.sub(r'\.+','',text)
    #ハングル除去
    text = re.sub(r'[가-힣]','',text)
    #繰り返し文字をまとめる
    text = re.sub(r"(.)\1{2,}", "\g<1>", text)
    #英文除去
    if checkAlnum(text):
        text = ''
    #ストップワード除去tf-idf?
    return text

def PreProcessing(file):
    new_file = []
    with open(file,'r',encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for text in csv_reader:
            new_file.append(preprocessing(text[0]))
    
    #何故か必要
    keep_file = []
    for i in new_file:
        if len(i) != 0:
            keep_file.append(i)

    preprocessing_file = 'pre_'+file
    with open(preprocessing_file,'w',encoding='utf-8',newline='')as f:
        writer = csv.writer(f)
        #writer.writerows(preprocessing_file)
        for line in keep_file:
            writer.writerow([line.strip()])
    
    return preprocessing_file

