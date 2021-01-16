import os


JSON_MODEL_PATH = 'model/model_json.json'
MODEL_PATH = 'model/model.h5'
MODEL_WEIGHTS_PATH = 'model/model_weights.h5'
QUESTONS_PATH = 'model/questions.txt'
ANSWERS_PATH = 'model/answers.txt'

FB_TOKEN = os.environ.get('FB_TOKEN')
FB_WEBHOOKS_VERIFICATION_TOKEN = os.environ.get('FB_WEBHOOKS_VERIFICATION_TOKEN')
FB_API_URL = f'https://graph.facebook.com/v5.0/me/messages?access_token={FB_TOKEN}'
