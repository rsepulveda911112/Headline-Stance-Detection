import os
from common.reader import JSONLineReader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

map_all = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
map_related = {'agree': 1, 'disagree': 1, 'discuss': 1, 'unrelated': 0}
map_stance = {'agree': 0, 'disagree': 1, 'discuss': 2}


def load_all_data(file_in, type_class, features):
    jsonl_reader = JSONLineReader()
    datas = jsonl_reader.read(os.getcwd() + file_in)
    df_in = pd.DataFrame(datas)
    label = 'label'

    if type_class == 'related':
        df_in[label] = df_in[label].replace('agree', 'related')
        df_in[label] = df_in[label].replace('disagree', 'related')
        df_in[label] = df_in[label].replace('discuss', 'related')

    elif type_class == 'stance':
        df_in = df_in[df_in[label] != 'unrelated']
        df_in = df_in.reset_index(drop=True)

    df_in[label] = labelencoder.fit_transform(df_in[label])
    l = df_in[label].to_list()
    h = df_in['sentence1'].to_list()
    p = df_in['sentences2'].to_list()
    f = []
    if features:
        df = pd.DataFrame(df_in[features[0]].to_list(), columns=['0'])
        for i in range(1,len(features)):
            df[str(i)] = df_in[features[i]].to_list()
        f = df.iloc[:,:].values
        f = list(f)
    else:
        f = [0] * len(h)
    list_of_tuples = list(zip(h, p, l, f))
    return pd.DataFrame(list_of_tuples, columns=['text_a', 'text_b', 'labels', 'feature'])