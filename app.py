from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

def clean_and_save_csv(file):
    # Load the dataset
    data = pd.read_csv(file)
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data.dropna(subset=['Date'], inplace=True)
    return data

def add_features(data):
    data['Lag1'] = data['Close'].shift(1)
    data['Lag2'] = data['Close'].shift(2)
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
    data.dropna(inplace=True)
    return data

def train_and_predict(data):
    features = ['Lag1', 'Lag2', 'MA5', 'MA20', 'MACD']
    target = 'Close'
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    latest_features = X.iloc[-1].to_frame().T
    predicted_price = model.predict(latest_features)[0]

    return {"predicted_price": predicted_price}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded."})
        file = request.files['file']
        data = clean_and_save_csv(file)
        data = add_features(data)
        result = train_and_predict(data)
        return jsonify(result)

    # Return HTML directly
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload CSV</title>
    </head>
    <body>
        <h1>Upload Your CSV File for Prediction</h1>
        <form action="/" method="post" enctype="multipart/form-data">
            <label for="file">Choose your CSV file:</label><br><br>
            <input type="file" name="file" id="file" accept=".csv" required><br><br>
            <button type="submit">Submit</button>
        </form>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)
