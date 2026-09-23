from fastapi import FastAPI
import pandas as pd
import pickle
from pydantic import BaseModel

# -------------------------------------------------
# Create FastAPI app
# -------------------------------------------------
app = FastAPI()

#-------------------------------------------------
# Set Base Model
#-------------------------------------------------

class LoanData(BaseModel):
    AGE: int
    EMPLOY: int
    ADDRESS: int
    DEBTINC: float
    CREDDEBT: float
    OTHDEBT: float


#-------------------------------------------------
#  Load the trained model + column order 
#-------------------------------------------------

with open('model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

train_columns = list(rf_model.feature_names_in_)

#--------------------------------------------------
# Define functions
#--------------------------------------------------


def load_test_data():
    test_data = pd.read_csv('BANK LOAN_TEST.csv')

    return test_data


def prepare_test_data(test_df):
    test_df['AGE'] = test_df['AGE'].astype("category")
    test_df = pd.get_dummies(test_df, columns=['AGE'], drop_first=True)

    y_test = test_df['DEFAULTER']
    X_test = test_df.drop(columns=['DEFAULTER', 'SN'])
    X_test = X_test.reindex(columns=train_columns, fill_value=0)

    return X_test, y_test, test_df

def get_prob_data(X_test, test_df):
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    result = test_df[['SN']].copy()
    result['Prob_Default'] = y_prob.round(4)
    result['Predicted_Default'] = y_pred

    return result
  

# -------------------------------------------------
# Health check endpoint
# -------------------------------------------------
@app.get("/health")
def health_check():

    return {"message": "Loan default prediction app is running"}


# ---------------------------------------------------------
# Prediction GET endpoint - for retrieving test predictions
# ---------------------------------------------------------
@app.get("/predict")
def test_data_analysis():
    test_data = load_test_data()
    X_test, _, test_df = prepare_test_data(test_data)
    prediction_data = get_prob_data(X_test, test_df)
    
    return prediction_data.to_dict(orient="records")

# -------------------------------------------------
# Prediction POST endpoint - for single submission
# -------------------------------------------------
@app.post("/predict")
def individual_data_analysis(loan: LoanData):
    row = pd.DataFrame([{
        'AGE': loan.AGE, 'EMPLOY': loan.EMPLOY, 'ADDRESS': loan.ADDRESS,
        'DEBTINC': loan.DEBTINC, 'CREDDEBT': loan.CREDDEBT, 'OTHDEBT': loan.OTHDEBT,
    }])
    row = pd.get_dummies(row, columns=['AGE'], drop_first=True)
    row = row.reindex(columns=train_columns, fill_value=0)

    prob = rf_model.predict_proba(row)[:, 1][0]
    predicted = int(prob > 0.5)

    return {"probability_default": round(float(prob), 4), "predicted_default": predicted}
