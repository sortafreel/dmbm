import json
from datetime import datetime, timezone

from dotenv import dotenv_values
from telethon import TelegramClient
from telethon.tl.types import Message

CONFIG = dotenv_values(".env")
WAR_START = datetime(2022, 2, 24, 0, 40, 0, 0, tzinfo=timezone.utc)
# Lviv Air Alerts channel
CHANNEL_ID = -1001399934598
# # Air Alerts UA channel
# CHANNEL_ID = -1766138888
MESSAGES = []


def prepare_message(post: Message) -> dict:
    return {
        "datetime": post.date.strftime("%d/%m/%Y, %H:%M:%S"),
        "message": post.message,
    }


async def collect_messages(channel_id: int) -> None:
    # Fill Telegram cache
    await client.get_dialogs()

    # Get the required channel
    alerts_channel = await client.get_entity(channel_id)

    async for post in client.iter_messages(alerts_channel):
        if isinstance(post.message, list):
            print("*" * 500)
            print(post.message)
            continue
        if post.date < WAR_START:
            break
        MESSAGES.append(prepare_message(post))


client = TelegramClient("name", int(CONFIG["APP_API_ID"]), CONFIG["APP_API_HASH"])
with client:
    client.loop.run_until_complete(collect_messages(CHANNEL_ID))

with open(f"exports/{CHANNEL_ID}-channel-messages.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(MESSAGES))
    print(f"Exported {len(MESSAGES)} messages.")
