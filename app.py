from flask import Flask, send_file
from flask_restx import Api, Resource, reqparse, fields
from flask_cors import CORS
import requests
import json
import threading
import random
import test_model


app = Flask(__name__)
CORS(app)
api = Api(app, version='1.0', title='ConnectX 문서', description='ConnectX API 문서', doc="/api-docs")

get_next_action_ns = api.namespace('get_next_action', description='다음 action 조회')

def get_next_action_func(file_content):
    state_list = list(file_content)
    model_name = 'DQNmodel_CNN'
    return test_model.test_main(state_list, model_name)

Modelstate = get_next_action_ns.model('get_next_action', strict=True, model={
    'list': fields.List(fields.List(fields.Integer), title='현재 state', default=[[0,0,0,0,0,0,0], [0,0,0,0,0,2,0], [0,1,0,0,2,1,0], [0,2,0,0,1,1,0], [1,2,2,2,1,1,2], [1,2,2,1,1,2,1]], required=True),
})

@get_next_action_ns.route('/')
class Test(Resource):
    @get_next_action_ns.expect(Modelstate)
    def post(self):
        return (get_next_action_func(api.payload['list'])), 200

if __name__ == '__main__':
    app.run(debug=True)
