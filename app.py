from flask import Flask, request, jsonify
import functools
import pandas as pd
from src_margins.core import calculate
from flasgger import Swagger
from utilities import convert_to_dataframe
import os
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

# Secret key to verify API key/token
SECRET_API_KEY = "margin-simulator-key"

# Simple function to check if the provided API key is correct
def api_key_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        # Get API key from headers
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != SECRET_API_KEY:
            return jsonify({'message': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/calculate_margin', methods=['POST'])
@api_key_required
def process_data():
    """
    A Method to calculate the margin
    ---
    summary: Calculate margin for given data
    description: This endpoint processes input data to calculate the margin for specified date ranges.
    tags:
      - Margin Calculation
    parameters:
      - in: body
        name: body
        required: true
        description: Input data in JSON format
        schema:
          type: object
          properties:
            dateFrom:
              type: string
              example: "2024-11-01"
            dateTo:
              type: string
              example: "2024-11-20"
            portfolios:
              type: object
              example:
                p1:
                  isin1: {"qt": 0, "tp": 23.189, "currency": "EUR"}
                  isin2: {"qt": 1, "tp": 23.189, "currency": "EUR"}
                p2:
                  isin1: {"qt": 2, "tp": 23.189, "currency": "EUR"}
    responses:
      200:
        description: A list of calculated margins for each portfolio and date
        schema:
          type: object
          additionalProperties:
            type: array
            items:
              type: object
              properties:
                portfolioNumber:
                  type: string
                  example: "P1"
                ExpectedShortFall:
                  type: number
                  format: float
                  example: 0.1
                DecoRelation:
                  type: number
                  format: float
                  example: 0.0
                whatIf:
                  type: number
                  format: float
                  example: 244.75
                markToMarket:
                  type: number
                  format: float
                  example: 494
                initialMargin:
                  type: number
                  format: float
                  example: 494.13
                grossPositionValue:
                  type: number
                  format: float
                  example: 494
                marginPercentage:
                  type: number
                  format: float
                  example: 100.02
    """
    # Get the JSON payload from the incoming request
    inputData = request.get_json()
    # Check if the data was provided
    if not inputData:
        return jsonify({"message": "No JSON payload provided"}), 400

    df = convert_to_dataframe(inputData)
    output = calculate(df, inputData["dateFrom"], inputData["dateTo"])
    # output = json.loads(output)
    
    portfoliosDates = {}
    
    for date in output:
        portfolios = []    
        outputJson = json.loads(output[date])
        for portofolio in outputJson["portfolio_nb"]:
            portfolioValue = {
                "portfolioNumber": outputJson["portfolio_nb"][portofolio],
                "ExpectedShortFall": round(outputJson["ES"][portofolio], 1),
                "DecoRelation": round(outputJson["DECO"][portofolio], 1),
                "whatIf":  round(outputJson["whatif"][portofolio], 1),
                "markToMarket":  round(outputJson["mtm"][portofolio], 1),
                "initialMargin":  round(outputJson["initial_margin"][portofolio], 1),
                "grossPositionValue":  round(outputJson["gross_pos_value"][portofolio], 1),
                "marginPercentage":  round(outputJson["margin_%"][portofolio], 1)
            }
            portfolios.append(portfolioValue)
        portfoliosDates[date] = portfolios
    return portfoliosDates

@app.route('/get_instruments', methods=['GET'])
def get_instruments():
    """
    Retrieve a list of instruments.
    ---
    summary: Retrieve instruments
    description: This API fetches a list of instruments available in the system.
    tags:
      - Instruments
    responses:
      200:
        description: Successfully retrieved the list of instruments.
        content:
          application/json:
            example: []
    """
    return None

@app.route('/get_portfolios', methods=['GET'])
def get_portfolio():
    """
    Retrieve portfolios information.
    ---
    summary: Get portfolios
    description: This API retrieves portfolios details.
    tags:
      - Portfolio
    responses:
      200:
        description: Successfully retrieved the portfolios details.
        content:
          application/json:
            example: null
    """
    return None

@app.route('/analytics', methods=['POST'])
def analytics():
    """
    Perform analytics on the input data.
    ---
    summary: Analytics computation
    description: This API performs analytics computations based on the provided input data.
    tags:
      - Analytics
    parameters:
      - in: body
        name: body
        required: true
        description: Input data for analytics in JSON format
        schema:
          type: object
          properties:
            dateFrom:
              type: string
              example: "2024-11-01"
            dateTo:
              type: string
              example: "2024-11-20"
            portfolios:
              type: object
              example:
                p1:
                  isin1: {"qt": 0, "tp": 23.189, "currency": "EUR"}
                  isin2: {"qt": 1, "tp": 23.189, "currency": "EUR"}
                p2:
                  isin1: {"qt": 2, "tp": 23.189, "currency": "EUR"}
    responses:
      200:
        description: Successfully performed analytics computations.
        content:
          application/json:
            example: null
    """
    return None

if __name__ == '__main__':
    app.run(debug=True)
