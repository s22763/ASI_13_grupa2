from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
import pandas as pd
import json
import psycopg2
import settings as s
from sqlalchemy import create_engine

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/model_download")
async def model():
    return FileResponse("model.pkl", filename="model.pkl", media_type="application/octet-stream")

@app.get("/download")
async def download():
    conn = psycopg2.connect(host=s.DATABASE_HOST,
                            database=s.DATABASE_NAME,
                            user=s.DATABASE_USER,
                            password=s.DATABASE_PASSWORD,
                            port=s.DATABASE_PORT)
    sql_query = 'SELECT * FROM health_data'
    df = pd.read_sql(sql_query, conn)
    conn.close()
    return df.to_dict()

@app.post("/preprocess")
async def preprocess(data: dict):
    X_train_json = data["X_train"]
    X_test_json = data["X_test"]
    X_validate_json = data["X_validate"]

    X_train = pd.DataFrame.from_records(json.loads(X_train_json))
    X_test = pd.DataFrame.from_records(json.loads(X_test_json))
    X_validate = pd.DataFrame.from_records(json.loads(X_validate_json))

    db = create_engine(s.DATABASE_CONSTRING)

    X_train.to_sql('train', db, if_exists='replace', index=False)
    X_test.to_sql('test', db, if_exists='replace', index=False)
    X_validate.to_sql('validate', db, if_exists='replace', index=False)
    return Response()

@app.post("/write_synthetic_data")
async def write_synthetic_data(data: dict):
    synthetic_data_json = data["synthetic_data"]
    synthetic_data = pd.DataFrame.from_records(json.loads(synthetic_data_json))
    db = create_engine(s.DATABASE_CONSTRING)
    synthetic_data.to_sql('health_data', db, if_exists='replace', index=False)
    return Response()