"""
将项目接入到telegram
"""
import asyncio
import os
import telebot
import urllib.parse
import requests
import json


bot = telebot.TeleBot('')

@bot.message_handler(commands=['start'])
def start_message(message):
    # bot.reply_to(message, '你好!')
    bot.send_message(message.chat.id, '你好，我是李博文教授。')


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    try:
        encoded_text = urllib.parse.quote(message.text)
        response = requests.post('http://localhost:8000/chat?query='
                                 +encoded_text, timeout=100)
        if response.status_code == 200:
            aiResponse = json.loads(response.text)
            if "msg" in aiResponse:
                bot.reply_to(message, aiResponse["msg"]["output"])
                audio_path = f"./speeches/{aiResponse['id']}.mp3"
                asyncio.run(check_audio(message, audio_path))
            else:
                bot.reply_to(message, "对不起，我不知道怎么回答你")
    except requests.RequestException as e:
        bot.reply_to(message, "对不起，请求出错了")


async def check_audio(message, audio_path):
    while True:
        if os.path.exists(audio_path):
            with open(audio_path, "rb") as f:
                bot.send_audio(message.chat.id, f)
            os.remove(audio_path)
            break
        else:
            print("waiting speech generation")
            await asyncio.sleep(1)


bot.infinity_polling()
