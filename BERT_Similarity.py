import csv
from transformers import BertTokenizer, BertModel
import torch

# 任意の5つの文章をリストに格納
texts = [
    "ヒカキンさんがラーメンが好きなのは夢のよう。hikakinの努力と商品には凄い尊敬を感じます。",
    "ヒカキンさんがコンビニで0味噌ラーメンを食べた報告、嬉しいですね！美味しかったと聞いてよかったです。",
    "ヒカキンさんがラーメンの夢を見て、その姿に尊敬します。0キンラーメンが売り切れるほど、hikakinの人気は嬉しいですね！",
    "ヒカキンさんが夢でまずいラーメンを食べたけど、実際は美味しかったとyoutubeで報告してくれました。彼の努力とすごい成果には尊敬します！",
    "hikakinさんが絶対に楽しみにしているセブンでの0ラーメンの発売、夢のように美味しいと聞いて大好きになりました！"
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
    with open('BERT_output_LDA.csv', 'w', newline='', encoding = 'utf-8') as outfile:
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
