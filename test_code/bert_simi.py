from transformers import BertTokenizer, BertModel
import torch

# 文章の例
text1 = "ヒカキン、絶対好き！発売の味噌ラーメン、楽しみ！大好きな味噌、Hikakinと一緒だ！"
#text2 = "ラーメンに凄くこだわって作られたのが伝わってきます発売日楽しみです"
#text2 = "美味しそう絶対買う"
text2 = "関係ない文章を入力した場合の類似度はどうなるのかの検証用文章"

# BERTのトークナイザーとモデルのロード
tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')

# トークナイズとテンソルへの変換
inputs1 = tokenizer(text1, return_tensors="pt")
inputs2 = tokenizer(text2, return_tensors="pt")

# BERTモデルを使用して埋め込みを取得
with torch.no_grad():
    outputs1 = model(**inputs1)
    outputs2 = model(**inputs2)

# 埋め込みの平均を取得
embeddings1 = outputs1.last_hidden_state.mean(dim=1)
embeddings2 = outputs2.last_hidden_state.mean(dim=1)

# コサイン類似度の計算
cos_sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)

print(cos_sim.item())
