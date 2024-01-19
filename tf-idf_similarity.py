from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import MeCab
import csv

def tokenize_japanese(text):
    mecab = MeCab.Tagger(r'-Owakati -d "C:\mecab-ipadic-neologd"')
    return mecab.parse(text).strip()

# 任意の5つの文章をリストに入力
texts = [
    "ヒカキンさんがラーメンが好きなのは夢のよう。hikakinの努力と商品には凄い尊敬を感じます。",
    "ヒカキンさんがコンビニで0味噌ラーメンを食べた報告、嬉しいですね！美味しかったと聞いてよかったです。",
    "ヒカキンさんがラーメンの夢を見て、その姿に尊敬します。0キンラーメンが売り切れるほど、hikakinの人気は嬉しいですね！",
    "ヒカキンさんが夢でまずいラーメンを食べたけど、実際は美味しかったとyoutubeで報告してくれました。彼の努力とすごい成果には尊敬します！",
    "hikakinさんが絶対に楽しみにしているセブンでの0ラーメンの発売、夢のように美味しいと聞いて大好きになりました！"
]

# これらの文章のTF-IDFベクトルを計算
vectorizer = TfidfVectorizer(tokenizer=tokenize_japanese)
tfidf_matrix = vectorizer.fit_transform(texts)

# 入力CSVファイルを読み込み
with open('pre_misokin_original.csv', 'r', encoding = 'utf-8') as file:
    reader = csv.reader(file)

    # 出力CSVファイルを準備
    with open('tf-idf_output_LDA.csv', 'w', newline='', encoding = 'utf-8') as outfile:
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
