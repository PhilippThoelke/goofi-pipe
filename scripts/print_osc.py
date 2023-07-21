"""A simple OSC client that prints messages received from an OSC server."""
import argparse
from pythonosc import dispatcher
from pythonosc import osc_server
parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="127.0.0.1")
parser.add_argument("--port", type=int, default=5005)


def handler(address, *args):
    print(f"{address}: {args}")

if __name__ == "__main__":
    args = parser.parse_args()
    address = args.ip
    port = args.port
    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/*", handler)
    server = osc_server.ThreadingOSCUDPServer(
        (address, port), dispatcher)
    print(f"Serving on {server.server_address}")
    server.serve_forever()