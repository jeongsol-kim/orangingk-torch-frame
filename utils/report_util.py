import json 
import requests


def read_my_setting():
    try:
        with open('utils/slack_data.json', 'r') as f:
            data = json.load(f)
            return data['token'], data['channel']
    except:
        raise FileNotFoundError

def slack_send_msg(msg):
    token, channel = read_my_setting()

    requests.post("https://slack.com/api/chat.postMessage",
        headers={"Authorization": "Bearer "+token},
        data={"channel": channel,"text": msg})
    
if __name__ == '__main__':
    slack_send_msg('test message')