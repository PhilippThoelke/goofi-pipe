import time
from typing import Any, Dict, Tuple

import numpy as np
import serial
import serial.tools.list_ports

from goofi.data import DataType
from goofi.node import Node
from goofi.params import IntParam


class SerialStream(Node):
    ESC = b"\xDB"
    END = b"\xC0"
    ESC_END = b"\xDC"
    ESC_ESC = b"\xDD"

    def config_params():
        return {"serial": {"sfreq": IntParam(512, 128, 1000), "port": ""}, "common": {"autotrigger": True}}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def setup(self):
        if self.params.serial.port.value:
            self.ser = serial.Serial(self.params.serial.port.value, 115200, timeout=1)
        else:
            port = self.detect_serial_port()
            if port is None:
                self.ser = None
                raise RuntimeError("No serial port found.")

            print(f"Found serial port {port}")
            self.ser = serial.Serial(port, 115200, timeout=1)

        self.last_time = None
        self.last_sample = None

    def process(self) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        data_buf, time_buf = [], []

        packet = bytearray()
        start = time.time()
        while time.time() - start < 1 / self.params.common.max_frequency.value:
            c = self.ser.read()
            if c == self.END:
                # ignore empty packets
                if packet and len(packet) == 2:
                    value = packet[0] << 8 | packet[1]
                    current_time = time.time()
                    data_buf.append(value)
                    time_buf.append(current_time)
                packet = bytearray()
            elif c == self.ESC:
                c = self.ser.read()
                if c == self.ESC_END:
                    packet.append(0xC0)
                elif c == self.ESC_ESC:
                    packet.append(0xDB)
                else:
                    print("Invalid escape sequence")
            elif len(c) > 0:
                packet.append(c[0])

        if len(data_buf) == 0:
            # no data received
            return None

        if self.last_time is None:
            # first time, just store the data
            self.last_time = time_buf[-1]
            self.last_sample = data_buf[-1]
            return None

        # resample the data
        dt = 1 / self.params.serial.sfreq.value
        xs = np.arange(time_buf[0], time_buf[-1], dt)
        # TODO: make sure the time array is correct and we don't have discontinuities
        data = np.interp(xs, time_buf, data_buf, left=self.last_sample)

        self.last_time = time_buf[-1]
        self.last_sample = data[-1]

        meta = {"sfreq": self.params.serial.sfreq.value}
        return {"out": (data, meta)}

    def detect_serial_port(self, names=["Arduino", "Serial"]):
        # detect the serial port
        ports = serial.tools.list_ports.comports()
        for port in ports:
            for name in names:
                if name in port.description:
                    return port.device
        return None

    def serial_port_changed(self, value: str) -> None:
        # reinitialize the client
        self.setup()
