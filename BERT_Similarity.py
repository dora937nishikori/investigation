import csv
from transformers import BertTokenizer, BertModel
import torch

# 任意の5つの文章をリストに格納
texts = [
    "味噌汁に豚汁の具を適当に入れてみたら、巨神兵入りみたいに大迫力！ラピュタのロボット兵が作ったかのよう。ハイボールと一緒に最高！",
    "豚汁を作るのが好きで、今日は味噌と酒で味付け、ゴボウを加えました。灰汁を取るのは勉強になるー。ヒロミのように美味しい！",
    "リュウジのレシピで0から豚汁を作りました。大きな里芋とごぼうを使って、本当に美味しい！最高に美味し！",
    "リュウジのレシピで0回の失敗もなく、豚汁を料理しました。とても美味しくて、代わりにうまいと思ったほど美味しかった！",
    "リュウジの動画を見て、至高の豚汁レシピを作りました。料理が最高に美味しくて、笑いながら楽しんだほどです！"
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
    with open('LDA豚汁類似度BERT.csv', 'w', newline='', encoding = 'utf-8') as outfile:
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
