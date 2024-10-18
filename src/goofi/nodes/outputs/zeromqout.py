import numpy as np
import zmq

from goofi.data import Data, DataType
from goofi.node import Node


class ZeroMQOut(Node):
    def config_params():
        return {"zero_mq": {"address": "127.0.0.1", "port": 6543}}

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def setup(self):
        if not hasattr(self, "context"):
            self.context = zmq.Context()

        if hasattr(self, "socket"):
            try:
                self.socket.close()
            except Exception:
                pass

        # bind a publisher socket
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.bind(f"tcp://{self.params.zero_mq.address.value}:{self.params.zero_mq.port.value}")

    def process(self, data: Data):
        if data is None:
            return
        data = data.data.astype(np.float32)
        self.socket.send_pyobj(data)

    def zero_mq_address_changed(self, value):
        # TODO: make sure socket stuff only happens on the main thread
        self.setup()

    def zero_mq_port_changed(self, value):
        # TODO: make sure socket stuff only happens on the main thread
        self.setup()
