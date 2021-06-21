from flask import Flask
from recommendation import reco_api
from flask_restx import Resource, Api, reqparse
from flask import Blueprint, request, redirect, url_for, flash, jsonify
from tensorflow import keras
import numpy as np
import pickle
import pandas as pd


app = Flask(__name__)
api = Api(app)

# Mapping of product ids and product names
product_names = np.load('product_names.npy', allow_pickle='TRUE').item()
# Loaded from popular products model, this is default when new user is entered
popular_products = ['B00006IE7J', 'B00005249G', 'B00006IEEV', 'B00004Z5WK', 'B00002NDS3',
       'B00006JNJD', 'B00006IFAY', 'B00002NDRX', 'B00006IFI5', 'B00002NDRT']
# Load the model
model = keras.models.load_model('first_model')
# Load the encodings
user_pkl = open('user_encoder.pkl', 'rb')
item_pkl = open('item_encoder.pkl', 'rb')
user_encoder = pickle.load(user_pkl)
item_encoder = pickle.load(item_pkl)
items = item_encoder.classes_
user_pkl.close()
item_pkl.close()

@api.route('/recommend')
@api.param('user', 'User Id For Recommendation')
class Recommend(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            args = parser.parse_args()
            print("args", args)
            user = args['user']
            print('user = ', np.array([user]))
            user_encoded = user_encoder.transform([user]*len(items))
            item_encoded = item_encoder.transform(items)
            X = np.array([user_encoded, item_encoded]).T
            test = [X[:,0], X[:,1]]
            predictions = model.predict(test)
            top_ten = self.top_ten_predicted(items, predictions)
            return str(self.convert(top_ten))
        except:
            return str(self.top_ten_default())

    def top_ten_predicted(self, items, predictions):
        predictions_df = pd.DataFrame(predictions, columns=['ratings'])
        item_id_df = pd.DataFrame(items, columns = ['itemId'])
        item_id_df['ratings'] = predictions_df['ratings']
        top_10 = item_id_df.sort_values(by = ['ratings'], ascending=False)[:10]
        return top_10['itemId'].values

    def convert(self, item_ids):
        names = []
        for item in item_ids:
            names.append(product_names[item])
        return names

    def top_ten_default(self):
        return self.convert(popular_products)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
