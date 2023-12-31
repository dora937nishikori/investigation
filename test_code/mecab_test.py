import MeCab

"""MeCab + NEologd"""

"""MeCab + NEologd + 分かち書き"""
'好きなことをして生きていくのがyoutuber'
'本当に努力の塊でしかないhikakinさんが頂点で本当によかったぁ'
'日本では鬼滅の刃が人気です'
text = '日本では鬼滅の刃が面白くて人気です'
#形態素解析
#インスタンス

#m = MeCab.Tagger(r'-d "C:\mecab-ipadic-neologd"')
m = MeCab.Tagger()
parses = m.parse(text).split('\n') #改行区切り
morphs = [] #形態素解析
removed = [] #空白連結

for parse in parses:
    #空白区切り
    tmp = parse.split('\t')
    print(tmp)
    #EOS、空白行削除
    if len(tmp) > 1:
        morphs.append(parse.split('\t'))

#品詞抽出
for morph in morphs:
    if '名詞' in morph[1] or '形容詞' in morph[1]: 
        removed.append(morph[0])

print(' '.join(removed))

#分かち書き
tagger = MeCab.Tagger(r'-Owakati -d "C:\mecab-ipadic-neologd"') #分かち書きと辞書の指定を同時にやるだけ
print(tagger.parse(text))