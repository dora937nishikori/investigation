from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import MeCab
import csv

def tokenize_japanese(text):
    mecab = MeCab.Tagger(r'-Owakati -d "C:\mecab-ipadic-neologd"')
    return mecab.parse(text).strip()

# 任意の5つの文章をリストに入力
texts = [
    "ヒカキン、絶対好き！発売の味噌ラーメン、楽しみ！大好きな味噌、Hikakinと一緒だ！",
    "今日、セブンで店員から元気をもらった！ヒカキンのYouTube活動、ラーメンへの努力、Hikakin最高！",
    "ヒカキン、尊敬してる。Youtuberじゃなくてもラーメンへの努力を応援！絶対好き！",
    "ヒカキン報告！YouTubeでラーメン商品紹介、楽しみ！康平キン、好きなYoutuberだ！",
    "Hikakinがセブンイレブンのみそラーメン発注応援！普通のコンビニに行列、紹介さい！"
]

# これらの文章のTF-IDFベクトルを計算
vectorizer = TfidfVectorizer(tokenizer=tokenize_japanese)
tfidf_matrix = vectorizer.fit_transform(texts)

# 入力CSVファイルを読み込み
with open('pre_misokin_original.csv', 'r', encoding = 'utf-8') as file:
    reader = csv.reader(file)

    # 出力CSVファイルを準備
    with open('tf-idf_output1218.csv', 'w', newline='', encoding = 'utf-8') as outfile:
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
