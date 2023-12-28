import numpy as np
import pandas as pd
import dill as dill

with open("model/sber_auto_pipe.pkl", 'rb') as file:
    model = dill.load(file)

df = pd.read_csv("data/df_key_action.csv")
df_1 = df[df.y == 1].drop(columns='y')

def check():
    data_line = df_1.iloc[1194:1200, :]
    pred = model['model'].predict(data_line)

    print(pred)

def check_df():
    pred = model['model'].predict(df_1)

    print(np.where(pred == 1.0)[0])


if __name__ == '__main__':
    check_df()