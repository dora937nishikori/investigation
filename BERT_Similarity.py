import csv
from transformers import BertTokenizer, BertModel
import torch

# 任意の5つの文章をリストに格納
texts = [
    "ブッコローが有隣堂0階でジェットストリームのペン先のエッジとデザインに感動して文字書きしてるの見て、マジで感動したわ。",
    "極細のボールペンとしてジェットストリームは細いペンが好きな左利きにも愛用されていて、インクもパイロットのものが最高！",
    "ブッコローと岡崎が愛用するボールペンとシャーペンの動画シリーズ、マジで面白い！キャラの気持ちが伝わってくる。",
    "個人的には、細いインクと大きなクリップが好きで、アクロのミリジェットは最高！",
    "ジェットストリームとアクロボールのボールペンは社長も愛用しているけど、ペン先のエッジが細くてトラブルが0円で解決するなんて、インクの質もさすが！"
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
with open('pre_ボールペン元コメント.csv', 'r', encoding = 'utf-8') as file:
    reader = csv.reader(file)

    # 出力CSVファイルを準備
    with open('BTMボールペン類似度BERT.csv', 'w', newline='', encoding = 'utf-8') as outfile:
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
