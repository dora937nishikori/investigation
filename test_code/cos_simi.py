from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import MeCab

def tokenize_japanese(text):
    mecab = MeCab.Tagger(r'-Owakati -d "C:\mecab-ipadic-neologd"')
    return mecab.parse(text).strip()

text1 = "ヒカキン、絶対好き！発売の味噌ラーメン、楽しみ！大好きな味噌、Hikakinと一緒だ！"
#text2 = "ラーメンに凄くこだわって作られたのが伝わってきます発売日楽しみです"
#text2 = "美味しそう絶対買う"
text2 = "関係ない文章を入力した場合の類似度はどうなるのかの検証用文章"

vectorizer = TfidfVectorizer(tokenizer=tokenize_japanese)
tfidf_matrix = vectorizer.fit_transform([text1, text2])

cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
print(cos_sim[0][0])
