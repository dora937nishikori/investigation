import csv
import pprint
words = ['ラーメン', 'てる', 'ヒカキン', 'とか', '本当に', '好き', '尊敬', '思う', 'すごい', '思い', 'ラーメン', '絶対', '味噌', 'とか', 'けど', 'ヒカキン', '買い', '食べ', 'おめでとう', 'hikakin', '食べ', 'たい', 'ラ ーメン', '楽しみ', '絶対', '買い', 'セブン', '発売', 'ので', 'hikakin', 'hikakin', 'ヒカキン', 'けど', 'ラーメン', 'てる', '本当に', '努力', '尊敬', '思う', '好き', 'ラーメン', 'ヒカキン', 'けど', 'てる', '本当に', 'hikakin', '楽しみ', 'とか', 'すぎ', '絶対']

file = 'hikakin_ramen\\pre_youtube-comments-list.csv'

similarity_sentence = []
with open(file,"r",encoding="utf-8")as f:
    reader = csv.reader(f)
    for sentence in reader:
        count = 0
        for word in words:
            if word in sentence[0]:
                count += 1
        similarity_sentence.append([count,sentence[0]])
pprint.pprint(similarity_sentence)

new_file = "hikakin_ramen\\frequency_similarity.csv"
with open(new_file,"w",encoding="utf-8",newline='')as f:
    writer = csv.writer(f)
    for i in similarity_sentence:
        writer.writerow(i)