from fastapi import FastAPI, WebSocket
from pyvirtualcam import Camera

app = FastAPI()


from obswebsocket import obsws, requests

# Адрес и порт для подключения к серверу OBS
host = "localhost"
port = 8082

# Пароль для подключения к серверу OBS (если есть)
password = ""

# Создаем экземпляр класса obsws и подключаемся к серверу OBS
ws = obsws(host, port, password)
ws.connect()

# Имя сцены из которой нужно получить изображение
scene_name = "Сцена"

# Запрашиваем текущий список источников сцены
sources = ws.call(requests.GetSceneItemList(scene_name))

# Находим источник, который отображает изображение (обычно это источник типа "Видеозахват")
source = next((source for source in sources.getSources() if source.getType() == "input"), None)

# Если источник найден, то получаем изображение
if source:
    # Запрашиваем изображение в текущем состоянии источника
    img = ws.call(requests.GetSourceScreenshot(source["name"], width=1920, height=1080, format="raw"))

    # Возвращаем массив байтов с изображением
    bytesss = img.to_bytes()

    print(bytesss)

# @app.websocket("/stream")
# async def stream(websocket: WebSocket):
#     # cam = Camera(width=640, height=480, fps=30)
#     # cam.start()
#     try:
#         while True:
#             # Получение ссылки на стрим
#             stream_url = cam.capture()
#             # Отправка ссылки на стрим на frontend при помощи WebSocket
#             await websocket.send_text(123123123)
#     finally:
#         cam.stop()