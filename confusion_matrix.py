import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# CSVファイルの読み込み
df = pd.read_csv('Misokin\TF-IDF90%.csv', header=None)
y_true = df[0]  # 正解ラベル一列目
y_pred = df[1]  # 予測ラベル二列目

# 混同行列の計算
conf_matrix = confusion_matrix(y_true, y_pred, labels=[1, 0])
print("Confusion Matrix:")
print(conf_matrix)

# 正解率、適合率、再現率、F値の計算
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
