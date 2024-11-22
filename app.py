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
    responses:
      200:
        description: A successful response
        content:
          application/json:
            example: {"message": "Hello, World!"}
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
                "ExpectedShortFall": outputJson["ES"][portofolio],
                "DecorRelation": outputJson["DECO"][portofolio],
                "whatIf": outputJson["whatif"][portofolio],
                "markToMarket": outputJson["mtm"][portofolio],
                "initialMargin": outputJson["initial_margin"][portofolio],
                "grossPositionValue": outputJson["gross_pos_value"][portofolio],
                "marginPercentage": outputJson["margin_%"][portofolio]
            }
            portfolios.append(portfolioValue)
        portfoliosDates[date] = portfolios
    return portfoliosDates

if __name__ == '__main__':
    app.run(debug=True)
