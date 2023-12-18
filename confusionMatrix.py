import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# CSVファイルのパス
actual_data_path = 'Misokin\正解データ.csv'
predicted_data_path = 'Misokin\BERT予測データ.csv'

# CSVファイルからデータを読み込む
actual_data = pd.read_csv(actual_data_path)
predicted_data = pd.read_csv(predicted_data_path)

# テキストを基にデータをマージする
merged_data = pd.merge(actual_data, predicted_data, on='テキスト', suffixes=('_actual', '_predicted'))

# 実際のラベルと予測ラベルを取得
actual_labels = merged_data['actual']
predicted_labels = merged_data['predicted']

# 混同行列と各種指標の計算
cm = confusion_matrix(actual_labels, predicted_labels)
accuracy = accuracy_score(actual_labels, predicted_labels)
precision = precision_score(actual_labels, predicted_labels)
recall = recall_score(actual_labels, predicted_labels)
f1 = f1_score(actual_labels, predicted_labels)

# 結果の出力
print("混同行列:", cm)
print("精度 (Accuracy):", accuracy)
print("適合率 (Precision):", precision)
print("再現率 (Recall):", recall)
print("F値 (F1 Score):", f1)
