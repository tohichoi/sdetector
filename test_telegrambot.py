from telegram.ext import Updater
import logging
from telegram.ext import CommandHandler
from io import BytesIO


def monitor(update, context):
    bio = BytesIO()
    bio.name='20200522T002004.mp4'

    fn='detected/20200522T002004.mp4'
    context.bot.send_video(chat_id=update.effective_chat.id,
        video=open(fn, 'rb'))
        


def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="순심이 탐지기에 오신걸 환영합니다")


if __name__ == '__main__':
    token=''
    with open('bot_token.txt') as fd:
        token=fd.readline().strip()

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)

    updater = Updater(token=token, use_context=True)
    dispatcher = updater.dispatcher

    start_handler = CommandHandler('start', start)
    dispatcher.add_handler(start_handler)

    monitor_handler = CommandHandler('monitor', monitor)
    dispatcher.add_handler(monitor_handler)

    updater.start_polling()
    updater.idle()
