from typing import Any, List, Tuple

from oscpy.client import send_bundle

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import IntParam


class OSCOut(Node):
    def config_input_slots():
        return {"data": DataType.TABLE}

    def config_params():
        return {"osc": {"address": "localhost", "port": IntParam(8000, 0, 65535), "prefix": "/goofi"}}

    def process(self, data: Data):
        if data is None or len(data.data) == 0:
            return

        # convert the data to a list of OSC messages
        messages = generate_messages(data, self.params.osc.prefix.value)

        # send the data as an OSC bundle
        send_bundle(messages, self.params.osc.address.value, self.params.osc.port.value)


def generate_messages(data: Data, prefix: str = "") -> List[Tuple[bytes, List[Any]]]:
    messages = []
    for key, val in data.data.items():
        # generate the OSC address
        addr = sanitize_address(prefix + "/" + key)

        if val.dtype == DataType.ARRAY:
            # convert the array to a list
            assert val.data.ndim < 2, "Numerical arrays must at most be one-dimensional."
            val = val.data.tolist()
        elif val.dtype == DataType.STRING:
            # simply use the string
            val = val.data
        elif val.dtype == DataType.TABLE:
            # recursively convert the table to a list of messages
            messages.extend(generate_messages(val, addr))
            continue
        else:
            raise ValueError(f"Unsupported data type {val.dtype} for OSC output.")

        # oscpy expects the message to be a list
        if not isinstance(val, list):
            val = [val]

        # add the message to the list
        messages.append((addr.encode(), val))

    return messages


def sanitize_address(address: str) -> str:
    """
    Sanitize an OSC address. This function removes leading and trailing slashes and replaces multiple slashes with a
    single slash.

    ### Parameters
    `address` : str
        The OSC address to sanitize.

    ### Returns
    str
        The sanitized OSC address.
    """
    return "/" + "/".join(a for a in address.split("/") if len(a) > 0).strip("/")
