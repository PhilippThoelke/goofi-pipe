import asyncio
import base64
import threading
import time
from io import BytesIO

import cv2
import numpy as np
import websockets
from PIL import Image

curr_img = None
trgt_img = None


def decode_image(msg):
    return np.array(Image.open(BytesIO(base64.b64decode(msg))))


async def server(websocket, _):
    while True:
        # Receive message
        msg = await websocket.recv()
        print("message received")

        # Decode message
        global curr_img, trgt_img
        new_img = decode_image(msg).astype(np.float32) / 255.0
        if trgt_img is None:
            trgt_img = np.array(new_img)
            curr_img = np.array(new_img)
        else:
            trgt_img[:] = np.array(new_img)


def display_img(alpha=0.97):
    global curr_img, trgt_img
    while True:
        # interpolate between prev_img and trgt_img
        if trgt_img is None:
            time.sleep(0.5)
            continue

        curr_img[:] = np.array(alpha * curr_img + (1 - alpha) * trgt_img)
        cv2.imshow("img", curr_img)
        cv2.waitKey(50)


threading.Thread(target=display_img).start()

start_server = websockets.serve(
    server, "localhost", 5105, ping_timeout=None, ping_interval=None
)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
