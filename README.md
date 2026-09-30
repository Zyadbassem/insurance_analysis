# Medical Insurance Cost Prediction

An end-to-end machine learning project that predicts medical insurance
charges from basic patient information.

The project uses a scikit-learn pipeline for preprocessing and
polynomial regression, then exposes the trained model through a FastAPI
REST API.

## Features

-   Categorical feature encoding with `OneHotEncoder`
-   Degree-2 polynomial feature generation
-   Linear regression
-   Train/test split
-   Serialized ML pipeline with `joblib`
-   FastAPI prediction endpoint
-   Pydantic request validation
-   CORS configuration for frontend integration

## Dataset

The model uses `insurance.csv`.

### Features

  Feature      Description
  ------------ ---------------------------------
  `age`        Patient age
  `sex`        Patient sex
  `bmi`        Body Mass Index
  `children`   Number of children
  `smoker`     Whether the patient is a smoker
  `region`     Residential region

### Target

-   `charges` --- medical insurance charges

## Machine Learning Pipeline

``` text
Raw data
   ↓
Train / test split
   ↓
ColumnTransformer
   ↓
OneHotEncoder
   ↓
PolynomialFeatures(degree=2)
   ↓
LinearRegression
   ↓
Serialized pipeline
```

The categorical features (`sex`, `smoker`, and `region`) are encoded
automatically by the pipeline.

The fitted preprocessing steps and regression model are saved together,
so the API does not need to manually reproduce the training
preprocessing.

## Project Structure

``` text
medical-insurance-prediction/
├── data/
│   └── insurance.csv
├── pipeline.py
├── main.py
├── model.pkl
├── frontend/
└── README.md
```

## Installation

Using pip:

``` bash
pip install pandas scikit-learn fastapi uvicorn joblib
```

Using uv:

``` bash
uv add pandas scikit-learn fastapi uvicorn joblib
```

## Train the Model

Run:

``` bash
python pipeline.py
```

This trains the pipeline using the training portion of the dataset and
saves the fitted pipeline as:

``` text
model.pkl
```

## Run the API

Start the FastAPI server:

``` bash
uvicorn main:app --reload
```

The API will normally be available at:

``` text
http://127.0.0.1:8000
```

Interactive API documentation:

``` text
http://127.0.0.1:8000/docs
```

## API Endpoints

### `GET /`

Health check endpoint.

Example response:

``` json
{
  "message": "Insurance API is running. Navigate to /docs to test it."
}
```

### `POST /charge`

Predicts the expected insurance charge for a patient.

Example request:

``` json
{
  "age": 35,
  "children": 2,
  "sex": "male",
  "smoker": "no",
  "region": "southeast",
  "bmi": 28.5
}
```

Example response:

``` json
{
  "Prediction": 12345.67
}
```

The actual prediction depends on the trained model.

## Input Validation

The API validates incoming data using Pydantic.

-   `age`: 18--120
-   `children`: 0 or greater
-   `bmi`: 10--70
-   `sex`: `male` or `female`
-   `smoker`: `yes` or `no`
-   `region`: `southwest`, `southeast`, `northwest`, or `northeast`

## Frontend Integration

The frontend can send a `POST` request to `/charge` with the patient's
information as JSON.

The backend returns the predicted insurance charge, which can then be
displayed by the frontend.

CORS is configured for local development and the deployed frontend.

## Technologies

-   Python
-   Pandas
-   Scikit-learn
-   FastAPI
-   Pydantic
-   Joblib
-   Uvicorn
-   JavaScript / frontend framework

## Disclaimer

This project is for educational purposes. The predicted insurance charge
is a model estimate and should not be treated as an actual insurance
quote, medical advice, or financial advice.

## Future Improvements

-   Add MSE and R² results to the documentation
-   Compare linear and polynomial regression performance
-   Add automated API tests
-   Deploy the FastAPI backend
-   Deploy the frontend
-   Add model/version tracking
