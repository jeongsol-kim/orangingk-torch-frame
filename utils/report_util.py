import os
from pathlib import Path
import getpass
import json 
import requests

cut_fn = lambda msg: msg.split('. ')[0].split('.\n')[0]
current_project = Path(os.getcwd()).stem
prompt = lambda msg: f"[{getpass.getuser()}/{current_project}]: {cut_fn(msg)}."

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
        data={"channel": channel,"text": prompt(msg)})
    
if __name__ == '__main__':
    slack_send_msg('test message')
