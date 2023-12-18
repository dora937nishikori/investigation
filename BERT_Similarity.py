import csv
from transformers import BertTokenizer, BertModel
import torch

# 任意の5つの文章をリストに格納
texts = [
    "ヒカキン、絶対好き！発売の味噌ラーメン、楽しみ！大好きな味噌、Hikakinと一緒だ！",
    "今日、セブンで店員から元気をもらった！ヒカキンのYouTube活動、ラーメンへの努力、Hikakin最高！",
    "ヒカキン、尊敬してる。Youtuberじゃなくてもラーメンへの努力を応援！絶対好き！",
    "ヒカキン報告！YouTubeでラーメン商品紹介、楽しみ！康平キン、好きなYoutuberだ！",
    "Hikakinがセブンイレブンのみそラーメン発注応援！普通のコンビニに行列、紹介さい！"
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
    with open('BERT_output1218.csv', 'w', newline='', encoding = 'utf-8') as outfile:
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
