from Preprocessing import PreProcessing
from Morphological import Morphological
from stopword_remove import Stopword_Remover

#元ファイル
file = 'ボールペン元コメント.csv'
#前処理
preprocessing = PreProcessing(file)

#分かち書き
morphological = Morphological(preprocessing)

#ストップワード除去
Stopword_Remover(morphological)

