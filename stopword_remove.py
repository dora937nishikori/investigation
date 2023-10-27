from ja_stopword_remover.remover import StopwordRemover
import pprint

stopwordRemover = StopwordRemover()

def Stopword_Remover(file):
    keeplist = []

    with open(file,'r',encoding='utf-8') as f:
        datalist = f.readlines()
        for data in datalist:
            keeplist.append(data.split())

    result_list = stopwordRemover.remove(keeplist)

    stopword_remove_file = 'remove_'+file
    with open(stopword_remove_file,'w',encoding='utf-8',newline='')as f:
        for row in result_list:
            if not row:
                continue
            words = ' '.join(row)
            f.write(words+'\n')