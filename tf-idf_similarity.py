from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import MeCab
import csv

def tokenize_japanese(text):
    mecab = MeCab.Tagger(r'-Owakati -d "C:\mecab-ipadic-neologd"')
    return mecab.parse(text).strip()

# 任意の5つの文章をリストに入力
texts = [
    "今日、リュウジの料理動画を見て豚汁のレシピを作りました！最高に美味しくて、美味しいです！",
    "リュウジのレシピで作った豚汁、美味しい！生姜とニンニクが効いた味噌汁、本当に美味しくて最高！",
    "今日、リュウジのレシピで豚汁を作りました。至高の味で、大好きな料理シリーズがまた一つ増えました！美味しかった！",
    "リュウジ兄さんの動画を見て料理した豚汁、家族にも大好評！本当に美味しいレシピでした。最高です！",
    "豚汁にゴボウ、大根、ネギをたっぷり使い、味噌と白だしで味付け。生姜とニンニクで風味豊かな料理になりました！"
]

# これらの文章のTF-IDFベクトルを計算
vectorizer = TfidfVectorizer(tokenizer=tokenize_japanese)
tfidf_matrix = vectorizer.fit_transform(texts)

# 入力CSVファイルを読み込み
with open('pre_豚汁元コメント.csv', 'r', encoding = 'utf-8') as file:
    reader = csv.reader(file)

    # 出力CSVファイルを準備
    with open('豚汁類似度TF-IDF.csv', 'w', newline='', encoding = 'utf-8') as outfile:
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
