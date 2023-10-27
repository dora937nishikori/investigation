import MeCab

"""MeCab + NEologd"""

"""MeCab + NEologd + 分かち書き"""
CONTENT = 'どうでもいいけど0時間で0万回再生はやべえな'
#分かち書き
#tagger = MeCab.Tagger(r'-Owakati -d "C:\mecab-ipadic-neologd"') #分かち書きと辞書の指定を同時にやるだけ

#形態素解析
tagger = MeCab.Tagger(r'-d "C:\mecab-ipadic-neologd"')
parse = tagger.parse(CONTENT)
print(parse)
