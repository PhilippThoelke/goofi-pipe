import time
import numpy as np
import serial
import serial.tools.list_ports
from goofi.node import Node
from goofi.data import DataType
from goofi.params import IntParam, BoolParam
from typing import Any, Dict, Tuple


class SerialStream(Node):
    def config_params():
        return {
            "serial": {"sfreq": IntParam(1000, 128, 1000), "auto_select": BoolParam(True), "port": "COM4"},
            "common": {"buffer_seconds": IntParam(5)},
        }

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def setup(self):
        self.last_processed_time = None

    def read_serial(self):
        serial_buffer = []
        buffer_times = []
        if self.params.serial.auto_select.value:
            ser = serial.Serial(self.detect_serial_port(), 115200, timeout=1)
        else:
            ser = serial.Serial(self.params.serial.port.value, 115200, timeout=1)
        print("SERIAL PORT OPENED", ser.name)
        ESC = b"\xDB"
        END = b"\xC0"
        ESC_END = b"\xDC"
        ESC_ESC = b"\xDD"

        packet = bytearray()
        while True:
            c = ser.read()
            if c == END:
                if packet:  # ignore empty packets
                    if len(packet) == 2:
                        value = packet[0] << 8 | packet[1]
                        current_time = time.time()
                        serial_buffer.append(value)
                        buffer_times.append(current_time)
                        break
                    else:
                        print("Invalid packet received")
                    packet = bytearray()
            elif c == ESC:
                c = ser.read()
                if c == ESC_END:
                    packet.append(0xC0)
                elif c == ESC_ESC:
                    packet.append(0xDB)
                else:
                    print("Invalid escape sequence")
            elif len(c) > 0:
                packet.append(c[0])
        return serial_buffer, buffer_times

    def process(self) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        data, times = self.read_serial()
        data = np.array(data)
        times = np.array(times)
        # Additional logic and processing.

        meta = {"sfreq": self.params.serial.sfreq.value, "dim0": ["serial"]}
        return {"out": (data, meta)}  # Here, data is your processed data, modify as needed.

    def detect_serial_port(self, name="Arduino"):
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if name in port.description or "Serial" in port.description:
                return port.device
        return None
