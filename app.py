# Flask app init

# Standard libraries
import os
import requests

# Third party libraries
from flask import Flask, request
from flask_cors import CORS

from keras.models import model_from_json
from tensorflow import keras

# Local libraries
from chatbot import do_predict
from constants import (
    FB_API_URL,
    FB_WEBHOOKS_VERIFICATION_TOKEN,
    JSON_MODEL_PATH,
    MODEL_PATH,
    MODEL_WEIGHTS_PATH
)

APP = Flask(__name__)
CORS(APP)

JSON_MODEL = open(JSON_MODEL_PATH, 'r')
JSON_MODEL_CONTENT = JSON_MODEL.read()
JSON_MODEL.close()
TRAINED_MODEL = model_from_json(JSON_MODEL_CONTENT)
TRAINED_MODEL.load_weights(MODEL_WEIGHTS_PATH)


@APP.route('/')
def entrypoint():
    return '<h1>ChatBot app entrypoint. Please notice that model path is /chatbot</h1>'


@APP.route('/test/<word>')
def test(word: str = None):
    res = '-'
    if word:
        res = do_predict(TRAINED_MODEL, str(word))
    return f'<h1>{res}</h1>'


@APP.route('/chatbot', methods=['GET'])
def webhook():
    """ Facebook webhooks flow is:
    1. Visitor sends a message on the messenger to the chatbot
    2. Facebook Messenger Platform fires an event
    3. Webhook notified with the event
    4. Our backend recieve the event and returns an HTTP response
    """
    token = request.args.get('hub.verify_token')
    if token == FB_WEBHOOKS_VERIFICATION_TOKEN:
        return request.args.get('hub.challenge')
    return 'Invalid authorization'


@APP.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    message = data['entry'][0]['messaging'][0]['message']
    sender_id = data['entry'][0]['messaging'][0]['sender']['id']
    if 'text' in message and message['text']:
        request_body = {
            'recipient': {
                'id': sender_id
            },
            'message': {
                'text': do_predict(TRAINED_MODEL, str(message['text']))
            }
        }
        response_message = requests.post(
           FB_API_URL,
           json=request_body
        ).json()
    else:
        response_message = 'ok'

    return response_message


if __name__ == '__main__':
    APP.run()
