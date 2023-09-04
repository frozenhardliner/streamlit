from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from ml import Forecasted_Algorithms
import logging
import os
import subprocess
import jwt, datetime

# # Disable FastAPI's loggers
# logging.getLogger("uvicorn").setLevel(logging.CRITICAL)
# logging.getLogger("uvicorn.access").setLevel(logging.CRITICAL)
# logging.getLogger("fastapi").setLevel(logging.CRITICAL)

# # Configure logging
# logging.basicConfig(level=logging.ERROR)
# logger = logging.getLogger(__name__)

# Database or some safe storage should be used for storing credentials.

valid_credentials = {
    "123": "123",
    "12": "12"
}

app = FastAPI()



# FastAPI exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": str(exc.detail)},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": str(exc)},
    )




# Define a model for the authentication request payload
class AuthRequest(BaseModel):
    username: str
    password: str

# Define a model for the data request payload
class DataRequest(BaseModel):
    file_content: str
    date_column: str
    regressor_columns: list
    products_column: str
    forecast_column: str
    forecast_year: Optional[int] = None
    forecast_month: Optional[int] = None
    user_prediction: str
    products_to_forecast: str

SECRET_KEY = "qwerty12345678asdfg"
def create_jwt_token(username: str) ->str:
    payload = {
        "sub": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=50),
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token


@app.post("/authenticate", tags=["Authentication"])
async def authenticate(request: AuthRequest):
    if request.username in valid_credentials and request.password == valid_credentials[request.username]:
        token = create_jwt_token(request:AuthRequest)
        return {"authenticated": True, "token": token}  # Return a "token"
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")


# Route for processing data
@app.post("/output", tags=["Data Processing"])
async def output(request: DataRequest):
    try:
        file_content = request.file_content
        date_column = request.date_column
        regressor_columns = request.regressor_columns
        products_column = request.products_column
        forecast_column = request.forecast_column
        forecast_year = request.forecast_year
        forecast_month = request.forecast_month
        user_prediction = request.user_prediction
        products_to_forecast = request.products_to_forecast
        # Perform data processing operations using the selected columns
        # If Forecasted_Algorithms supports async, use 'await' here.
        result = Forecasted_Algorithms(file_content, date_column, regressor_columns, products_column, forecast_column,user_prediction, forecast_year, forecast_month, products_to_forecast )

        # selected_columns = {
        #     "file_content": file_content,
        #     "date_column": date_column,
        #     "regressor_columns": [regressor_columns],
        #     "products_column": products_column,
        #     "forecast_column": forecast_column,
        #     "predict_year": forecast_year,
        #     "user_prediction": user_prediction
        # }

        return result 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# Terminate existing processes on port 8888
try:
    subprocess.run(["taskkill", "/F", "/PID", "1000"], check=True)
except subprocess.CalledProcessError:
    # Handle if no process is found with the given PID
    pass

# Run the FastAPI application with Uvicorn on port 8080
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501, reload=False)
