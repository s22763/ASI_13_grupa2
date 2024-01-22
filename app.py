import streamlit as st
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
import os
import urllib.request
import pickle
import pandas as pd
import requests as r

bootstrap_project(os.getcwd())
session = KedroSession.create()
urla = "http://127.0.0.1:8000/model_download"
model = pickle.load(urllib.request.urlopen(urla))
params = session.load_context().params

st.write("co ja robie")

if st.button("run pipeline"):
    session.run()    
    
if st.button("load new model"):
    model = pickle.load(urllib.request.urlopen(urla))

if st.button("generate synthetic data"):
    response = r.request("GET", "http://127.0.0.1:8000/download").json()
    df = pd.DataFrame.from_dict(response)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    model = GaussianCopulaSynthesizer(metadata)
    model.fit(df)
    synthetic_data = model.sample(len(df))

    r.post("http://localhost:8000/write_synthetic_data", json={
        "synthetic_data": synthetic_data.to_json(orient="records")
    })

gender = st.text_input("gender")
age = st.text_input("age")
hypertension = st.text_input("hypertension")
heart_disease = st.text_input("heart_disease") 
smoking_history = st.text_input("smoking history")
bmi = st.text_input("bmi")
hba1c_level = st.text_input("hba1c_level")
blood_glucose_level = st.text_input("blood_glucose_level")

pred = None
diabetResult = "nie wiadomo czy jest chory"

if st.button("predict"):
    d = {"gender": [gender], "age": [age], "hypertension": [hypertension], "heart_disease": [heart_disease], "smoking_history": [smoking_history],
            "bmi": [bmi], "hba1c_level": [hba1c_level], "blood_glucose_level": [blood_glucose_level]}
    df = pd.DataFrame(data=d)
    pred = model.predict(df)[0]
    if pred == 1:
        diabetResult = "jest chory"
    elif pred == 0:
        diabetResult = "jest nie chory"

st.header(diabetResult)