import csv
import pprint

correct_file = 'Accuracy_evaluation\\misokin\\correct_middle.csv'
similarity_file = 'Accuracy_evaluation\\misokin\\30words_middle_similarity.csv'
#frequency_file = 'Accuracy_evaluation\\frequency_data.csv'

#アノテーションした正解データ
def correct(file):
    correct = []
    with open(file,"r",encoding='utf-8')as f:
        reader = csv.reader(f)
        for sentence in reader:
            correct.append(sentence[0])
    return correct

#類似度上位データ
def similarity(file):
    similarity = []
    with open(file,"r",encoding='utf-8')as f:
        reader = csv.reader(f)
        for sentence in reader:
            similarity.append(sentence[0])
    return similarity

#単語頻度上位データ
def frequency(file):
    frequency = []
    with open(file,"r",encoding='utf-8')as f:
        reader = csv.reader(f)
        for sentence in reader:
            frequency.append(sentence[0])
    return frequency

#精度計算
def precision(correct_data,evaluation_data):
    precision_count = 0
    for correct_sentence in correct_data:
        for evaluation_sentence in evaluation_data:
            if correct_sentence == evaluation_sentence:
                precision_count += 1
                break

    return precision_count/len(evaluation_data)

ans = precision(correct(correct_file),similarity(similarity_file))
print(ans)