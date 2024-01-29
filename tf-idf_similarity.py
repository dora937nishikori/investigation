from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import MeCab
import csv

def tokenize_japanese(text):
    mecab = MeCab.Tagger(r'-Owakati -d "C:\mecab-ipadic-neologd"')
    return mecab.parse(text).strip()

# 任意の5つの文章をリストに入力
texts = [
    "味噌汁に豚汁の具を適当に入れてみたら、巨神兵入りみたいに大迫力！ラピュタのロボット兵が作ったかのよう。ハイボールと一緒に最高！",
    "豚汁を作るのが好きで、今日は味噌と酒で味付け、ゴボウを加えました。灰汁を取るのは勉強になるー。ヒロミのように美味しい！",
    "リュウジのレシピで0から豚汁を作りました。大きな里芋とごぼうを使って、本当に美味しい！最高に美味し！",
    "リュウジのレシピで0回の失敗もなく、豚汁を料理しました。とても美味しくて、代わりにうまいと思ったほど美味しかった！",
    "リュウジの動画を見て、至高の豚汁レシピを作りました。料理が最高に美味しくて、笑いながら楽しんだほどです！"
]

# これらの文章のTF-IDFベクトルを計算
vectorizer = TfidfVectorizer(tokenizer=tokenize_japanese)
tfidf_matrix = vectorizer.fit_transform(texts)

# 入力CSVファイルを読み込み
with open('pre_豚汁元コメント.csv', 'r', encoding = 'utf-8') as file:
    reader = csv.reader(file)

    # 出力CSVファイルを準備
    with open('LDA豚汁類似度TF-IDF.csv', 'w', newline='', encoding = 'utf-8') as outfile:
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
