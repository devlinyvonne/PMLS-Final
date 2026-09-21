from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# -------------------------------------------------
# Create FastAPI app
# -------------------------------------------------
app = FastAPI()


#--------------------------------------------------
# Define functions
#--------------------------------------------------


def load_data():
    train_data = pd.read_csv('BANK LOAN.csv')
    test_data = pd.read_csv('BANK LOAN_TEST.csv')

    return train_data, test_data

def prepare_data(train_df, test_df):
    # Convert Age to categorical column
    train_df['AGE'] = train_df['AGE'].astype("category")
    test_df['AGE'] = test_df['AGE'].astype("category")

    # Getting dummies for AGE
    train_df = pd.get_dummies(train_df, columns=['AGE'], drop_first=True)
    test_df = pd.get_dummies(test_df, columns=['AGE'], drop_first=True)
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

    # Assign dpendent/independent variables
    y_train = train_df['DEFAULTER']
    X_train = train_df.drop(columns=['DEFAULTER', 'SN' ])
    y_test = test_df['DEFAULTER']
    X_test = test_df.drop(columns=['DEFAULTER', 'SN' ])

    return X_train, y_train, X_test, y_test

def get_prob_data(X_train, y_train, X_test, test_df):
    rf_model = RandomForestClassifier(
        n_estimators=500,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

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


# -------------------------------------------------
# Main endpoint
# -------------------------------------------------
@app.get("/predict")
def data_analysis():

    train_data, test_data = load_data()

    X_train, y_train, X_test, y_test = prepare_data(train_data, test_data)

    prediction_data = get_prob_data(X_train, y_train, X_test, test_data)

    return prediction_data.to_dict(orient="records")
