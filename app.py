# Flask app init

# Third party libraries
from flask import Flask, request
from flask_cors import CORS

# Local libraries
from chatbot import do_predict
from constants import (
    FB_API_URL,
    FB_WEBHOOKS_VERIFICATION_TOKEN
)

app = Flask(__name__)
CORS(app)


@app.route('/')
def entrypoint():
    return '<h1>ChatBot app entrypoint. Pls notice that model path is /cb</h1>'


@app.route('/cb', methods=['GET'])
def webhook():
    """ Facebook webhooks flow is:
    1. Visitor sends a message on the messenger to the chatbot
    2. Facebook Messenger Platform fires an event
    3. Webhook notified with the event
    4. Our backend recieve the event and returns an HTTP response
    """
    token = request.args.get('hub.verify_token')
    print(token)
    if token == FB_WEBHOOKS_VERIFICATION_TOKEN:
        return request.args.get('hub.challenge')
    return 'Invalid authorization'


@app.route('/cb', methods=['POST'])
def chatbot():
    request = request.get_json()
    message = request['entry'][0]['messaging'][0]['message']
    if message['text']:
        request_body = {
            'recipient': {
                'id': sender_id
            },
            'message': {
                'text': do_predict(str(message['text']))
            }
        }
        response = requests.post(
           FB_API_URL,
           json=request_body
        ).json()
        response_message = response
    else:
        response_message = 'ok'

    return response_message


if __name__ == '__main__':
    app.run(threaded=True, port=5000)
