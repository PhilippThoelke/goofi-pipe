import logging
from multiprocessing import Pipe, Process

from goofi.node import NodeRef
from goofi.nodes.add import Add
from goofi.nodes.constant import Constant

logger = logging.getLogger(__name__)


class Manager:
    def __init__(self) -> None:
        self.node_refs = dict()

    def create_node(self, node_class):
        node_pipe, manager_pipe = Pipe()
        node_process = Process(target=node_class, args=(manager_pipe,), daemon=True)
        node_process.start()
        node_ref = NodeRef(node_process, node_pipe)
        logger.info(f'Initialized node "{node_class.__name__}" with PID {node_ref.pid}')

        self.node_refs[node_ref.pid] = node_ref


if __name__ == "__main__":
    manager = Manager()

    # create a node
    manager.create_node(Constant)
    manager.create_node(Constant)
    manager.create_node(Add)

    import time

    time.sleep(10)
