from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

class Student_Data(BaseModel):
    hours: int
    prev_score: int
    extra_curr: bool
    sleep_hrs: int
    num_sample_paper: int

# Load the saved model from the pickle file
with open('stacking_regressor.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post("/")
async def pref_score(item: Student_Data):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    return { "hell0" : "World" }
