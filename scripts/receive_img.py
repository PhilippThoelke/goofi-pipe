import base64
import socket
import threading
import time
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

curr_img = None
trgt_img = None


def decode_image(msg):
    return np.array(Image.open(BytesIO(base64.b64decode(msg))))


def receive_image(host="localhost", port=8000):
    global curr_img, trgt_img
    while True:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        success = False
        while not success:
            try:
                client_socket.connect((host, port))
                success = True
            except socket.error:
                time.sleep(1)

        try:
            data = b""
            while True:
                part = client_socket.recv(1024)
                if not part:
                    client_socket.detach()
                    break
                data += part
        except socket.error:
            client_socket.detach()
            continue

        if len(data) == 0:
            print("len 0")
            time.sleep(0.1)
            continue

        new_img = decode_image(data).astype(np.float32) / 255.0

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


# Create threads for receiving and displaying images
thread_recv = threading.Thread(target=receive_image)
thread_disp = threading.Thread(target=display_img)

# Start the threads
thread_recv.start()
thread_disp.start()
