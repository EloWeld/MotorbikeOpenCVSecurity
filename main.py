import asyncio
import os
import cv2
import numpy as np
from telegram import Bot
from collections import deque

# Токен вашего Telegram бота
TOKEN = os.environ.get('BOT_TOKEN')
# Идентификатор чата, в который нужно отправлять сообщения
CHAT_ID = os.environ.get('CHAT_ID')

# Максимальный размер буфера
BUFFER_SIZE = 100


async def send_telegram_message(text):
    bot = Bot(token=TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=text)
    
    
def detect_objects_on_video():
    # Загрузка конфигурационного файла и весов для модели YOLO
    net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

    # Список классов объектов, которые может детектировать модель YOLO
    classes = []
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Индексы классов мотоцикла и человека
    motorcycle_class_id = classes.index('motorbike')
    person_class_id = classes.index('person')
    
    # Буферы для отслеживания кадров с мотоциклом и людьми
    motorcycle_buffer = deque(maxlen=BUFFER_SIZE)
    person_buffer = deque(maxlen=BUFFER_SIZE)

    # Загрузка видео с камеры
    video = cv2.VideoCapture(0)

    while True:
        # Чтение кадра из видеопотока
        ret, frame = video.read()

        if not ret:
            break

        # Используем предобученную модель YOLO для детекции объектов
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layer_outputs = net.forward(output_layers_names)

        # Отображение детектированных объектов на кадре
        class_ids = []
        confidences = []
        boxes = []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.1 and (class_id == motorcycle_class_id or class_id == person_class_id):
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.0, 0.1)
                
        people_detected = False
        motorcycle_detected = False
        for i in range(len(boxes)):
            if i in indexes:
                if class_ids[i] == motorcycle_class_id:
                    motorcycle_detected = True
                    
                elif class_ids[i] == person_class_id:
                    people_detected = True
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        motorcycle_buffer.append(motorcycle_detected)
        person_buffer.append(people_detected)

        cp = len([x for x in person_buffer if x])
        cm = len([x for x in motorcycle_buffer if x])
        # Отображение прогрессбаров
        progressbar_length = int(cm / len(motorcycle_buffer) * 20)
        progressbar_motorcycle = '[' + '#' * progressbar_length + ' ' * (20 - progressbar_length) + ']'
        cv2.putText(frame, progressbar_motorcycle, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        progressbar_length = int(cp / len(person_buffer) * 20)
        progressbar_people = '[' + '#' * progressbar_length + ' ' * (20 - progressbar_length) + ']'
        cv2.putText(frame, progressbar_people, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Отображение кадра с детектированными объектами
        cv2.imshow('Video', frame)
        
        # Проверка условий для отправки сообщения
        if cp > BUFFER_SIZE * 0.2 and len(person_buffer) > BUFFER_SIZE * 0.9:
            asyncio.run(send_telegram_message('Обнаружены люди около мотоцикла!'))

        if cm < BUFFER_SIZE * 0.5 and len(motorcycle_buffer) > BUFFER_SIZE * 0.9:
            asyncio.run(send_telegram_message('Мотоцикл отсутствует в большей части кадров!'))

        # Выход из цикла при нажатии клавиши 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    # Освобождение ресурсов
    video.release()
    cv2.destroyAllWindows()

# Запуск функции для детекции объектов на видео
detect_objects_on_video()