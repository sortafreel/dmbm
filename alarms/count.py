# coding: utf8
import json
from datetime import datetime

with open("result.json", encoding="utf-8") as f:
    messages = json.loads(f.read())["messages"]

alarms = {}
no_alarms = []
for message in messages:
    msg_datetime = datetime.strptime(message["date"], "%Y-%m-%dT%H:%M:%S")
    msg_index = msg_datetime.strftime("%Y-%m-%d_%H:%M")
    # print("*"*50)
    # print(message["date"])
    # print(msg_datetime)
    # print(msg_index)
    if type(message["text"]) is list:
        continue
    if "Повітряна тривога!" in message["text"]:
        alarms[msg_index] = {"text": message["text"], "datetime": msg_datetime}
    else:
        no_alarms.append(message["text"])

alarm_by_hour = {x: 0 for x in range(24)}
for alarm in alarms.values():
    alarm_by_hour[alarm["datetime"].hour] += 1

sorted_alarms = dict(
    sorted(alarm_by_hour.items(), key=lambda item: item[1], reverse=True)
)

alarms_by_part_of_the_day = {(0, 6): 0, (6, 12): 0, (12, 18): 0, (18, 24): 0}
for key, value in sorted_alarms.items():
    for period in alarms_by_part_of_the_day:
        if key >= period[0] and key < period[1]:
            alarms_by_part_of_the_day[period] += value
            break

print(alarms_by_part_of_the_day)
