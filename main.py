from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

# Initialize scaler
scaler = MinMaxScaler()
encoder = LabelEncoder()

def fit_preprocess():
    df = pd.read_csv(r"./students.csv")
    num_columns = [x for x in df.columns if x != 'Extracurricular Activities' and x != 'Performance Index']
    df[num_columns]  = scaler.fit_transform(df[num_columns])
    df['Extracurricular Activities'] = encoder.fit_transform(df['Extracurricular Activities'])

fit_preprocess()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
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


@app.post("/")
async def predict_performance(item: StudentData):
    # Create a DataFrame from the incoming JSON data
    extra = {
        "Extracurricular Activities": item.extracurricular_activities,
    }
    newData = {
        "Hours Studied": item.hours_studied,
        "Previous Scores": item.previous_scores,
        "Sleep Hours": item.sleep_hours,
        "Sample Question Papers Practiced": item.sample_question_papers_practiced
    }
    extraData = pd.DataFrame([extra])
    data = pd.DataFrame([newData])

    # Scale numerical columns
    # Convert 'extracurricular_activities' column to numeric
    extraData['Extracurricular Activities'] = extraData['Extracurricular Activities'].apply(lambda x: 1 if x.lower() == 'yes' else 0)

    # Transform data using the scaler
    scaled_data = scaler.transform(data)

    # Convert scaled_data to a DataFrame
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns.tolist())

    # Add the 'extraData' DataFrame after the first two columns of the 'data' DataFrame
    data = pd.concat([scaled_df.iloc[:, :2], extraData, scaled_df.iloc[:, 2:]], axis=1)

    # Make prediction
    prediction = model.predict(data)

    return {"predicted_performance": prediction[0]}