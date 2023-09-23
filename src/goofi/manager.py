import importlib
import time
from os import path
from typing import Dict, Optional

import yaml

from goofi.gui.window import Window
from goofi.message import Message, MessageType
from goofi.node_helpers import NodeRef, list_nodes


class NodeContainer:
    """
    The node container keeps track of all nodes in the manager. It provides methods to add and remove nodes,
    and to access them by name.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, NodeRef] = {}

    def add_node(self, name: str, node: NodeRef, force_name: bool = False) -> str:
        """
        Adds a node to the container with a unique name.

        ### Parameters
        `name` : str
            The name of the node.
        `node` : NodeRef
            The node to add.
        `force_name`: bool
            If True, raise an error if the name is already taken. Otherwise makes the name unique.
        """
        if not isinstance(name, str):
            raise ValueError(f"Expected string, got {type(name)}.")
        if not isinstance(node, NodeRef):
            raise ValueError(f"Expected NodeRef, got {type(node)}.")

        if force_name:
            # check if the name is already taken
            if name in self._nodes:
                raise KeyError(f"Node {name} already in container.")
            # register the node under the given name
            self._nodes[name] = node
            return name

        # generate a unique name for the node
        idx = 0
        while f"{name}{idx}" in self._nodes:
            idx += 1
        # register the node under the generated name
        self._nodes[f"{name}{idx}"] = node
        return f"{name}{idx}"

    def remove_node(self, name: str) -> None:
        """
        Terminates the node and removes it from the container.

        ### Parameters
        `name` : str
            The name of the node.
        """
        if name in self._nodes:
            self._nodes[name].terminate()
            del self._nodes[name]
            return True
        raise KeyError(f"Node {name} not in container")

    def __getitem__(self, name: str) -> NodeRef:
        return self._nodes[name]

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._nodes


class Manager:
    """
    The manager keeps track of all nodes, and provides methods to add and remove nodes, and links between them.
    It also interfaces with the GUI to display the nodes and links, and to handle user interaction.

    ### Parameters
    `headless` : bool
        Whether to run in headless mode. If `True`, the GUI will not be started.
    """

    def __init__(self, headless: bool = True) -> None:
        # preload all nodes to avoid delays
        # TODO: add proper logging
        print("Starting goofi-pipe...")
        # TODO: list_nodes can probably be more efficient
        list_nodes(verbose=True)

        # TODO: add proper logging
        print("Initializing goofi-pipe manager.")

        self._headless = headless
        self._running = True
        self.nodes = NodeContainer()

        if not self.headless:
            Window(self)

    def add_node(self, node_type: str, category: str, notify_gui: bool = True, name: Optional[str] = None) -> str:
        """
        Adds a node to the container.

        ### Parameters
        `node_type` : str
            The name of the node type (the node's class name).
        `category` : str
            The category of the node.
        `notify_gui` : bool
            Whether to notify the gui to add the node.
        `name` : Optional[str]
            Raises an error if the name is already taken. If `None`, a unique name is generated.

        ### Returns
        `name` : str
            The name of the node.
        """
        # TODO: add proper logging
        print(f"Adding node '{node_type}' from category '{category}'.")

        # import the node
        mod = importlib.import_module(f"goofi.nodes.{category}.{node_type.lower()}")
        node = getattr(mod, node_type)

        # instantiate the node and add it to the container
        ref = node.create_local()[0]
        name = self.nodes.add_node(node_type.lower(), ref)

        # add the node to the gui
        if not self.headless and notify_gui:
            Window().add_node(name, ref)
        return name

    def remove_node(self, name: str, notify_gui: bool = True) -> None:
        """
        Removes a node from the container.

        ### Parameters
        `name` : str
            The name of the node.
        `notify_gui` : bool
            Whether to notify the gui to remove the node.
        """
        # TODO: add proper logging
        print(f"Removing node '{name}'.")

        self.nodes.remove_node(name)
        if not self.headless and notify_gui:
            Window().remove_node(name)

    def add_link(self, node_out: str, node_in: str, slot_out: str, slot_in: str, notify_gui: bool = True) -> None:
        """
        Adds a link between two nodes.

        ### Parameters
        `node_out` : str
            The name of the output node.
        `node_in` : str
            The name of the input node.
        `slot_out` : str
            The output slot name of `node_out`.
        `slot_in` : str
            The input slot name of `node_in`.
        `notify_gui` : bool
            Whether to notify the gui to add the link.
        """
        # TODO: Prevent multiple links to the same input slot. The GUI already prevents this, but the manager should too.
        self.nodes[node_out].connection.send(
            Message(
                MessageType.ADD_OUTPUT_PIPE,
                {"slot_name_out": slot_out, "slot_name_in": slot_in, "node_connection": self.nodes[node_in].connection},
            )
        )

        if not self.headless and notify_gui:
            Window().add_link(node_out, node_in, slot_out, slot_in)

    def remove_link(self, node_out: str, node_in: str, slot_out: str, slot_in: str, notify_gui: bool = True) -> None:
        """
        Removes a link between two nodes.

        ### Parameters
        `node_out` : str
            The name of the output node.
        `node_in` : str
            The name of the input node.
        `slot_out` : str
            The output slot name of `node_out`.
        `slot_in` : str
            The input slot name of `node_in`.
        `notify_gui` : bool
            Whether to notify the gui to remove the link.
        """
        self.nodes[node_out].connection.send(
            Message(
                MessageType.REMOVE_OUTPUT_PIPE,
                {"slot_name_out": slot_out, "slot_name_in": slot_in, "node_connection": self.nodes[node_in].connection},
            )
        )

        if not self.headless and notify_gui:
            Window().remove_link(node_out, node_in, slot_out, slot_in)

    def terminate(self, notify_gui: bool = True) -> None:
        """
        Terminates the manager and all nodes.

        ### Parameters
        `notify_gui` : bool
            Whether to notify the gui to terminate.
        """
        if not self.headless and notify_gui:
            try:
                # terminate the gui, which calls manager.terminate() with notify_gui=False once it is closed
                Window().terminate()
                return
            except Exception:
                # TODO: add proper logging
                print("Closing the GUI failed.")

        # TODO: add proper logging
        print("Shutting down goofi-pipe manager.")
        # terminate the manager
        self._running = False
        for node in self.nodes:
            self.nodes[node].connection.send(Message(MessageType.TERMINATE, {}))
            self.nodes[node].connection.close()

    def save(self, filepath: Optional[str] = None, overwrite: bool = False, timeout: float = 3.0) -> None:
        """
        Saves the state of the manager to a file.

        ### Parameters
        `filepath` : Optional[str]
            The path to the file to save to. If `None`, a default filename is generated in the current directory.
        `overwrite` : bool
            Whether to overwrite an existing file.
        `timeout` : float
            The timeout in seconds for waiting for a response from each node.
        """
        # if no filepath was given, use a default filename in the current directory
        if not filepath:
            filepath = "."

        # make sure we get a string
        if not isinstance(filepath, str):
            raise ValueError(f"Expected string, got {type(filepath)}.")

        if path.exists(filepath) and path.isdir(filepath):
            # directory was given, create a default, non-conflicting filename
            idx = 0
            while path.exists(path.join(filepath, f"untitled{idx}.gfi")):
                idx += 1
            filepath = path.join(filepath, f"untitled{idx}.gfi")

        # add the file extension if it is missing
        if not filepath.endswith(".gfi"):
            filepath += ".gfi"

        # check if the file already exists
        if path.exists(filepath) and not overwrite:
            raise FileExistsError(f"File {filepath} already exists.")

        # TODO: add proper logging
        print(f"Saving manager state to '{filepath}'.")

        # request all nodes to serialized their state
        for name in self.nodes:
            self.nodes[name].serialize()

        # wait for all nodes to respond, i.e. their serialized_state is not None
        start = time.time()
        serialized_nodes = {}
        for name in self.nodes:
            while self.nodes[name].serialized_state is None and time.time() - start < timeout:
                # wait for the node to respond or for the timeout to be reached
                time.sleep(0.01)

            # check if we got a response in time
            if self.nodes[name].serialized_state is None:
                raise TimeoutError(f"Node {name} did not respond to serialize request.")
            serialized_nodes[name] = self.nodes[name].serialized_state

        # generate a list of links from the serialized nodes
        links = []
        for node_name_out, node in serialized_nodes.items():
            # iterate over all output slots of the current node
            for slot_name_out, conns in node["out_conns"].items():
                # iterate over all connections of the current slot
                for slot_name_in, conn in conns:
                    # find the node that matches the output connection of the current slot
                    for node_name_in in serialized_nodes.keys():
                        if conn == self.nodes[node_name_in].connection:
                            # found the node, add the link
                            links.append(
                                {
                                    "node_out": node_name_out,
                                    "node_in": node_name_in,
                                    "slot_out": slot_name_out,
                                    "slot_in": slot_name_in,
                                }
                            )
                            break
                    # NOTE: it's okay if we didn't find the node, it could be some external connection (e.g. GUI)

        # remove the output connections from the serialized_nodes dict so we can convert it to yaml
        for node in serialized_nodes.values():
            node.pop("out_conns")

        # convert the manager instance into yaml format
        manager_yaml = yaml.dump({"nodes": serialized_nodes, "links": links})

        # write the yaml to the file
        with open(filepath, "w") as f:
            f.write(manager_yaml)

    @property
    def running(self) -> bool:
        return self._running

    @property
    def headless(self) -> bool:
        return self._headless


def main(duration: float = 0, args=None):
    """
    This is the main entry point for goofi-pipe. It parses command line arguments, creates a manager
    instance and runs all nodes until the manager is terminated.

    ### Parameters
    `duration` : float
        The duration to run the manager for. If `0`, runs indefinitely.
    `args` : list
        A list of arguments to pass to the manager. If `None`, uses `sys.argv[1:]`.
    """
    import argparse
    import time

    # parse arguments
    parser = argparse.ArgumentParser(description="goofi-pipe")
    parser.add_argument("--headless", action="store_true", help="run in headless mode")
    args = parser.parse_args(args)

    # create manager
    manager = Manager(headless=args.headless)

    if duration > 0:
        # run for a fixed duration
        time.sleep(duration)
        manager.terminate()

    try:
        # run indefinitely
        while manager.running:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.terminate()


if __name__ == "__main__":
    main()
