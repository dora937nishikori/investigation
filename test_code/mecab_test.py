import MeCab

"""MeCab + NEologd"""

"""MeCab + NEologd + 分かち書き"""
text = 'ヒカキンには好感持てるけど'
#分かち書き
#tagger = MeCab.Tagger(r'-Owakati -d "C:\mecab-ipadic-neologd"') #分かち書きと辞書の指定を同時にやるだけ

#形態素解析
m = MeCab.Tagger(r'-d "C:\mecab-ipadic-neologd"')
nouns = [line[1] for line in m.parse(text).splitlines()
               if "名詞" in line.split()[-1]]
parse = m.parse(text)
print(m)
print(parse)
print(nouns)
