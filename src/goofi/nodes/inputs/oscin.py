import numpy as np
from oscpy.server import OSCThreadServer

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, IntParam, StringParam


class OSCIn(Node):
    def config_params():
        return {
            "osc": {
                "address": StringParam("0.0.0.0"),
                "port": IntParam(9000, 0, 65535),
                "keep_messages": BoolParam(True, doc="Keep all received messages"),
                "clear": BoolParam(trigger=True, doc="Clear all stored messages"),
            },
            "common": {"autotrigger": True},
        }

    def config_output_slots():
        return {"message": DataType.TABLE}

    def setup(self):
        self.server = OSCThreadServer(advanced_matching=True)
        self.server.listen(address=self.params.osc.address.value, port=self.params.osc.port.value, default=True)

        # bind to possible addresses of depth 10 (is there a better way to do this?)
        for i in range(1, 11):
            self.server.bind(b"/*" * i, self.callback, get_address=True)

        self.messages = {}

    def callback(self, address, *args):
        if len(args) > 1:
            raise ValueError(
                "For now the OSCIn node only support a single argument per message. "
                "Please open an issue if you need more (https://github.com/PhilippThoelke/goofi-pipe/issues)."
            )

        val = args[0]
        if isinstance(val, bytes):
            val = Data(DataType.STRING, val.decode(), {})
        else:
            val = Data(DataType.ARRAY, np.array([val]), {})

        self.messages[address.decode()] = val

    def process(self):
        if len(self.messages) == 0:
            return None

        data = self.messages

        if not self.params.osc.keep_messages.value or self.params.osc.clear.value:
            self.messages = {}

        return {"message": (data, {})}

    def osc_address_changed(self, address):
        self.server.stop_all()
        self.server.terminate_server()
        self.server.join_server()
        self.setup()

    def osc_port_changed(self, port):
        self.server.stop_all()
        self.server.terminate_server()
        self.server.join_server()
        self.setup()
