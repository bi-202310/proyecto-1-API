from typing import Optional
from fastapi import FastAPI
from dataModel import DataModel
import pandas as pd
from joblib import load

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

@app.post("/predict/bow-rf")
def predict_bow_rf(data: DataModel):
    df = pd.DataFrame(data.dict(), columns = data.dict().keys(), index=[0])
    df.columns = data.columns()
    model = load('assets/bow_rf.joblib')
    prediction = model.predict(df)
    return prediction

@app.post("/predict/tfidf-gb")
def predict_tfidf_gb(data: DataModel):
    df = pd.DataFrame(data.dict(), columns = data.dict().keys(), index=[0])
    df.columns = data.columns()
    model = load('assets/tfidf-gb.joblib')
    prediction = model.predict(df)
    return prediction

@app.post("/predict/tfidf-rf")
def predict_tfidf_rf(data: DataModel):
    df = pd.DataFrame(data.dict(), columns = data.dict().keys(), index=[0])
    df.columns = data.columns()
    model = load('assets/tfidf_rf.joblib')
    prediction = model.predict(df)
    return prediction

@app.post("/predict/tfidf-nb")
def predict_tfidf_nb(data: DataModel):
    df = pd.DataFrame(data.dict(), columns = data.dict().keys(), index=[0])
    df.columns = data.columns()
    model = load('assets/tfidf_nb.joblib')
    prediction = model.predict(df)
    return prediction