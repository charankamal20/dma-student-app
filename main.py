from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

app = FastAPI()
# Load the saved model from the pickle file
with open('stacking_regressor.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a Pydantic model for the data received from the user
class StudentData(BaseModel):
    hours_studied: int
    previous_scores: int
    extracurricular_activities: str
    sleep_hours: int
    sample_question_papers_practiced: int

# Define a function to preprocess the data
def preprocess_data(item: StudentData) -> pd.DataFrame:
    # Convert the Pydantic model to a dictionary
    data_dict = item.dict()

    # Create a DataFrame from the dictionary
    df = pd.DataFrame([data_dict])

    # Apply the same preprocessing steps as in your ML model
    scaler = MinMaxScaler()
    encoder = LabelEncoder()

    # Select numerical columns for scaling
    num_columns = [x for x in df.columns if x != 'extracurricular_activities']

    # Scale numerical columns
    df[num_columns] = scaler.fit_transform(df[num_columns])

    # Encode 'extracurricular_activities' column
    df['extracurricular_activities'] = encoder.fit_transform(df['extracurricular_activities'])

    return df

# Define the FastAPI route
@app.post("/")
async def pref_score(item: StudentData):
    # Preprocess the data
    preprocessed_data = preprocess_data(item)

    # Perform further processing or return the preprocessed data
    return {"preprocessed_data": preprocessed_data.to_dict()}