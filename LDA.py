import gensim
from gensim import corpora
import pandas as pd

def main():
    # テキストファイルの読み込み
    # ここではファイル名を 'wakachi_text.txt' と仮定します
    file_path = 'remove_pre_misokin_original.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        documents = file.readlines()

    # トークン化（各行が分かち書きされたテキスト）
    tokenized_docs = [doc.strip().split() for doc in documents]

    # 辞書の作成
    dictionary = corpora.Dictionary(tokenized_docs)

    # コーパスの作成
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    # LDAモデルの構築とトレーニング
    num_topics = 5  # トピック数
    num_words = 10  # 各トピックの上位単語数
    alpha = 0.01    # トピックの事前分布に対するパラメータ
    eta = 0.01      # 単語の事前分布に対するパラメータ

    lda_model = gensim.models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=15, alpha=alpha, eta=eta)

    #出力方法別
    df =pd.DataFrame()
    for t in range(num_topics):
        word=[]
        for i, prob in lda_model.get_topic_terms(t, topn=15):
            word.append(dictionary.id2token[int(i)])
        _ = pd.DataFrame([word],index=[f'topic{t+1}'])
        df = pd.concat([df, _], ignore_index = True, axis = 0)
    df_transposed = df.T
    print(df_transposed)

    '''
    # トピックと単語の表示
    topics = lda_model.print_topics(num_words=num_words)
    for topic in topics:
        print(topic)
    '''

if __name__ == '__main__':
    main()
