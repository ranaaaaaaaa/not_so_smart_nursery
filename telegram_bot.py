# 1) Creating a telegram bot using BotFather
# 2) Getting our telegram bot token
# 3) Getting our chat id
# 4) Sending message using Python

# make sure to /start your bot to work

TOKEN = "8893150300:AAHoDwaqYHTwOJusgo-9TeXY2HXBd1Dwa0Q"
CHAT_ID = '5949246928'

import requests

def send_alert():
    message = 'SERIOUS ALERT! CHECK ON YOUR BABY RIGHT NOW!'
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={message}"

    print(requests.get(url).json())
