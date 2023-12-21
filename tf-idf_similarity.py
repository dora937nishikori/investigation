from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import MeCab
import csv

def tokenize_japanese(text):
    mecab = MeCab.Tagger(r'-Owakati -d "C:\mecab-ipadic-neologd"')
    return mecab.parse(text).strip()

# 任意の5つの文章をリストに入力
texts = [
    "ヒカキンさんがラーメンが好きなのを知って、すごく嬉しいです！彼の元気と努力が絶対味噌ラーメンのように強い味を出していますね！",
    "ヒカキンさんが美味しいラーメンを楽しみにしているのを聞いて、応援する気持ちが強まります。みそラーメンを食べる報告、お願いしますね！ラーメン屋さんでの体験、楽しみにしています！",
    "ヒカキンさんがラーメン好きなのは、尊敬するYouTuberとしての素晴らしい一面です。彼の商品や活動に絶対注目して、報告を待っています。すごいですね！",
    "ヒカキンさんがラーメンを絶対楽しみにしているのを聞いて、私も大好きな味噌ラーメンが好きになりました。彼の新しい発売はいつもすごいですね！",
    "康平さんと奥村さんが休日の土日に外食でスガキヤの味噌ラーメンを食べたけど、残念ながらまずいと感じたようですね。お昼のちゃんねるでその話を聞きました。"
]

# これらの文章のTF-IDFベクトルを計算
vectorizer = TfidfVectorizer(tokenizer=tokenize_japanese)
tfidf_matrix = vectorizer.fit_transform(texts)

# 入力CSVファイルを読み込み
with open('pre_misokin_original.csv', 'r', encoding = 'utf-8') as file:
    reader = csv.reader(file)

    # 出力CSVファイルを準備
    with open('tf-idf_output1221.csv', 'w', newline='', encoding = 'utf-8') as outfile:
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
