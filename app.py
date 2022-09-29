from os.path import join

import uvicorn
##ASGI
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
# import ipy
import ipyparallel.serialize.codeutil
from pydantic import BaseModel
from sarcasm_detector import Concern
from sarcasm_model import SarcasmModel

app = FastAPI()


class Sentence(BaseModel):
    statement: str


pickle_in = open("sarcastic.pkl", "rb")
sarcasm_check = pickle.load(pickle_in)

pickle_out = open("sarcas.pkl", "rb")
result = pickle.load(pickle_out)


@app.get('/')
def index():
    return {'message': 'Hello, analyst'}


@app.get('/{name}')
def get_name(name: str):
    return {'Hello, ': f'{name} welcome to your pricing assistance', }


@app.post('/predict')
def sarcasm_predictor(data: Sentence):
    data = data.dict()
    statement = data['statement']

    prediction = sarcasm_check.predict_sarcasm(statement)

    return prediction


@app.post('/predictor')
def sarcasm_predict(data: Sentence):
    data = data.dict()
    statement = data['statement']

    ans = result.predict_sarcasm(statement)

    return ans


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.2', port=8000)
