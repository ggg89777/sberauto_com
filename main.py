import dill as dill
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()
with open("model/sber_auto_pipe.pkl", 'rb') as file:
    model = dill.load(file)

df_0 = pd.read_csv("data/df_key_action.csv")
df_1 = df_0[df_0.y == 1]
df_1 = df_1.drop(['y'], axis=1)



class Form(BaseModel):
    client_id: float
    visit_time: object
    visit_number: int
    hit_number: float
    hit_page_path: object
    hit_date: object
    hit_time: float
    utm_source: object
    utm_medium: object
    utm_campaign: object
    utm_adcontent: object
    utm_keyword: object
    device_category: object
    device_os: object
    device_brand: object
    device_screen_resolution: object
    device_browser: object
    geo_country: object
    geo_city: object


class Prediction(BaseModel):
    client_id: float
    pred: float


@app.get("/status")
def status():
    return "I’m OK"


@app.get("/version")
def version():
    return model["metadata"]


@app.post("/predict", response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame(form.dict(), index=[0])
    df = df.append(df_1)
    pred = model["model"].predict(df)

    return {
        "client_id": form.client_id,
        "pred": pred[0]
    }


def main():
    with open("model/sber_auto_pipe.pkl", 'rb') as file:
        model = dill.load(file)
        print(model["metadata"])