from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
import pickle
import os

query_columns = ['age', 'bmi', 'children', 'smoker_cond']
lr = pickle.load(open('../../models/lr.sav', 'rb'))

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)

@app.route('/')
def test_home():
    return 'Fiz uma API Bosta'

@app.route('/sentiment/<entry>')
@basic_auth.required
def sentiment(entry):
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(entry)
    return sentiment

# @app.route('/regression/<int:db>')
# def prediction(db):
#     reg = lr.predict([[db]])
#     return str(reg) 

@app.route('/regression3', methods=['POST'])
@basic_auth.required
def regressionAPI3params():
    data = request.get_json()
    data_input = [data[column] for column in query_columns]
    reg_3 = lr.predict([data_input])
    return jsonify(insurance_cost= round(reg_3[0], 2))



app.run(debug=True, host='0.0.0.0')