import csv
from transformers import BertTokenizer, BertModel
import torch

# 任意の5つの文章をリストに格納
texts = [
    "ヒカキンさんがラーメンが好きなのを知って、すごく嬉しいです！彼の元気と努力が絶対味噌ラーメンのように強い味を出していますね！",
    "ヒカキンさんが美味しいラーメンを楽しみにしているのを聞いて、応援する気持ちが強まります。みそラーメンを食べる報告、お願いしますね！ラーメン屋さんでの体験、楽しみにしています！",
    "ヒカキンさんがラーメン好きなのは、尊敬するYouTuberとしての素晴らしい一面です。彼の商品や活動に絶対注目して、報告を待っています。すごいですね！",
    "ヒカキンさんがラーメンを絶対楽しみにしているのを聞いて、私も大好きな味噌ラーメンが好きになりました。彼の新しい発売はいつもすごいですね！",
    "康平さんと奥村さんが休日の土日に外食でスガキヤの味噌ラーメンを食べたけど、残念ながらまずいと感じたようですね。お昼のちゃんねるでその話を聞きました。"
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
with open('pre_misokin_original.csv', 'r', encoding = 'utf-8') as file:
    reader = csv.reader(file)

    # 出力CSVファイルを準備
    with open('BERT_output1221.csv', 'w', newline='', encoding = 'utf-8') as outfile:
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
