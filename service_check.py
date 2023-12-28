import requests
import pandas as pd
import time


df = pd.read_csv("data/df_key_action.csv", keep_default_na=False)
df_1 = df[df.y == 1]
df_1 = df_1.drop(['y'], axis=1)
df_1['hit_time'] = df_1['hit_time'].apply(lambda x: 0.0 if x == "" else x)

def request_service(dfr):
    for i in range(0, len(dfr)):
        data_line = dfr.iloc[i]
        data_line = data_line.to_json()
        ping = requests.post('http://127.0.0.1:8000/predict', data=data_line)
        ping = ping.json()

        print(ping)

if __name__ == '__main__':
    request_service(df_1)