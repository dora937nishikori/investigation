from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import MeCab
import csv

def tokenize_japanese(text):
    mecab = MeCab.Tagger(r'-Owakati -d "C:\mecab-ipadic-neologd"')
    return mecab.parse(text).strip()

# 任意の5つの文章をリストに入力
texts = [
    "ブッコローが有隣堂0階でジェットストリームのペン先のエッジとデザインに感動して文字書きしてるの見て、マジで感動したわ。",
    "極細のボールペンとしてジェットストリームは細いペンが好きな左利きにも愛用されていて、インクもパイロットのものが最高！",
    "ブッコローと岡崎が愛用するボールペンとシャーペンの動画シリーズ、マジで面白い！キャラの気持ちが伝わってくる。",
    "個人的には、細いインクと大きなクリップが好きで、アクロのミリジェットは最高！",
    "ジェットストリームとアクロボールのボールペンは社長も愛用しているけど、ペン先のエッジが細くてトラブルが0円で解決するなんて、インクの質もさすが！"
]

# これらの文章のTF-IDFベクトルを計算
vectorizer = TfidfVectorizer(tokenizer=tokenize_japanese)
tfidf_matrix = vectorizer.fit_transform(texts)

# 入力CSVファイルを読み込み
with open('pre_ボールペン元コメント.csv', 'r', encoding = 'utf-8') as file:
    reader = csv.reader(file)

    # 出力CSVファイルを準備
    with open('BTMボールペン類似度TF-IDF.csv', 'w', newline='', encoding = 'utf-8') as outfile:
        writer = csv.writer(outfile)

        for row in reader:
            input_text = row[0]
            input_tfidf = vectorizer.transform([input_text])

            # 最大類似度の計算
            max_similarity = -1
            for i in range(len(texts)):
                cos_sim = cosine_similarity(input_tfidf, tfidf_matrix[i:i+1])
                max_similarity = max(max_similarity, cos_sim[0][0])

            # 出力CSVに書き込み
            writer.writerow([max_similarity, input_text])
