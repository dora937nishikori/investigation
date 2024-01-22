import csv
from transformers import BertTokenizer, BertModel
import torch

# 任意の5つの文章をリストに格納
texts = [
    "今日、リュウジの料理動画を見て豚汁のレシピを作りました！最高に美味しくて、美味しいです！",
    "リュウジのレシピで作った豚汁、美味しい！生姜とニンニクが効いた味噌汁、本当に美味しくて最高！",
    "今日、リュウジのレシピで豚汁を作りました。至高の味で、大好きな料理シリーズがまた一つ増えました！美味しかった！",
    "リュウジ兄さんの動画を見て料理した豚汁、家族にも大好評！本当に美味しいレシピでした。最高です！",
    "豚汁にゴボウ、大根、ネギをたっぷり使い、味噌と白だしで味付け。生姜とニンニクで風味豊かな料理になりました！"
]

# BERTのトークナイザーとモデルのロード
tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')

# 任意の文章の埋め込みを計算
embeddings = []
with torch.no_grad():
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        embeddings.append(embedding)

# 入力CSVファイルを読み込み
with open('pre_豚汁元コメント.csv', 'r', encoding = 'utf-8') as file:
    reader = csv.reader(file)

    # 出力CSVファイルを準備
    with open('豚汁類似度BERT.csv', 'w', newline='', encoding = 'utf-8') as outfile:
        writer = csv.writer(outfile)

        for row in reader:
            input_text = row[0]
            input_embedding = tokenizer(input_text, return_tensors="pt")
            output = model(**input_embedding)
            input_embedding = output.last_hidden_state.mean(dim=1)

            # 類似度の計算
            max_similarity = -1
            for embedding in embeddings:
                cos_sim = torch.nn.functional.cosine_similarity(input_embedding, embedding)
                max_similarity = max(max_similarity, cos_sim.item())

            # 出力CSVに書き込み
            writer.writerow([max_similarity, input_text])
