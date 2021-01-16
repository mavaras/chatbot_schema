import os


MODEL_PATH = 'model/training_model.h5'
FB_TOKEN = ''#os.environ['FB_TOKEN']
FB_WEBHOOKS_VERIFICATION_TOKEN = ''#os.environ['FB_WEBHOOKS_VERIFICATION_TOKEN']
FB_API_URL = f'https://graph.facebook.com/v5.0/me/messages?access_token={FB_TOKEN}'
