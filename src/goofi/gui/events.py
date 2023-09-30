import time
from copy import deepcopy

import dearpygui.dearpygui as dpg

from goofi.node_helpers import list_nodes

################################
######### Mouse Events #########
################################


def is_click_inside(item):
    """Check if the mouse click was inside the given item."""
    mouse_pos = dpg.get_mouse_pos(local=False)
    item_pos = dpg.get_item_pos(item)
    item_size = dpg.get_item_rect_size(item)
    return (
        mouse_pos[0] >= item_pos[0]
        and mouse_pos[1] >= item_pos[1]
        and mouse_pos[0] <= item_pos[0] + item_size[0]
        and mouse_pos[1] <= item_pos[1] + item_size[1]
    )


def click_callback(_, btn, win):
    """Update node selection after a click event."""
    selected = dpg.get_selected_nodes(win.node_editor)
    if len(selected) == 1:
        win._select_node(selected[0])
    else:
        win._select_node(None)

    if win.create_node_window is not None and not is_click_inside(win.create_node_window):
        # the create node window is open but the click was outside of it, close it
        escape(win)

    # TODO: do the same thing for the file selection window, which has a broken get_item_pos


def double_click_callback(_, btn, win):
    """If double click was left mouse button, open the window for creating a node."""
    if btn == 0 and len(dpg.get_selected_nodes(win.node_editor)) == 0:
        create_node(win)


################################
########## Key Events ##########
################################


def key_press_callback(_, data, win):
    """Handle key press events by calling the appropriate function from the key handler map."""
    if data in KEY_HANDLER_MAP:
        KEY_HANDLER_MAP[data](win)


#############################
########## Actions ##########
#############################


def delete_selected_item(win):
    """Deletes the selected node or link."""
    for node in dpg.get_selected_nodes(win.node_editor):
        win._remove_node(node)
    for link in dpg.get_selected_links(win.node_editor):
        win._remove_link(link)


def select_node_callback(sender, data, user_data):
    """Callback for when a node is selected in the create node window."""
    win, node = user_data
    pos = dpg.get_item_pos(win.create_node_window)
    # clean up the state of the GUI
    escape(win)
    dpg.clear_selected_nodes(win.node_editor)
    dpg.clear_selected_links(win.node_editor)
    # create the node inside the manager, which will notify the window
    win.manager.add_node(node.__name__, node.category(), pos=pos)


def create_node(win):
    """Opens the window for creating a node. If the window is already open, switch to the next tab."""

    # create a dictionary of nodes by category
    categories = {}
    for node in list_nodes():
        cat = node.category()
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(node)

    if win.create_node_window is not None:
        if dpg.is_item_focused(win.create_node_window):
            # the window is open and focused, switch to the next tab
            tab_bar = dpg.get_item_user_data(win.create_node_window)
            # increment tab index
            if dpg.is_key_down(dpg.mvKey_Shift):
                win.last_create_node_tab = (win.last_create_node_tab - 1) % len(categories)
            else:
                win.last_create_node_tab = (win.last_create_node_tab + 1) % len(categories)
            # switch to the next tab
            tab = f"tab_{list(categories.keys())[win.last_create_node_tab]}"
            dpg.set_value(tab_bar, tab)
            return

        # the window is open but not focused, close it to reopen later in this function
        dpg.delete_item(win.create_node_window)
        win.create_node_window = None

    if not dpg.is_item_hovered(win.node_editor):
        # the node editor is not hovered, do not open the window
        return

    # create a new window instance
    win.create_node_window = dpg.add_window(
        label="Create Node", pos=dpg.get_mouse_pos(local=False), no_collapse=True, autosize=True
    )

    # create a tab bar with a tab for each category
    with dpg.tab_bar(parent=win.create_node_window) as tab_bar:
        for cat, nodes in categories.items():
            with dpg.tab(label=cat, tag=f"tab_{cat}"):
                # create a button for each node in the category
                for node in nodes:
                    dpg.add_button(label=node.__name__, callback=select_node_callback, user_data=(win, node))

    # switch to the current tab
    dpg.set_value(tab_bar, f"tab_{list(categories.keys())[win.last_create_node_tab]}")
    # store the tab bar for switching tabs later
    dpg.set_item_user_data(win.create_node_window, tab_bar)


def escape(win):
    """
    Provides a way to cancel the creation of a node by closing the node selection window.
    If it was already closed, clears the current node and link selection.
    """
    if win.create_node_window is not None:
        dpg.delete_item(win.create_node_window)
        win.create_node_window = None
    elif win.file_selection_window is not None:
        dpg.delete_item(win.file_selection_window)
        win.file_selection_window = None
    else:
        dpg.clear_selected_nodes(win.node_editor)
        dpg.clear_selected_links(win.node_editor)


def save_manager(win):
    """Save the manager to a file. We go through the Window to potentially open a file selection dialog."""
    if dpg.is_key_down(dpg.mvKey_Control):
        win.save()


def copy_selected_nodes(win, timeout: float = 0.1):
    """Copy the selected nodes to the clipboard."""
    if not dpg.is_key_down(dpg.mvKey_Control):
        return

    # retrieve selected nodes and their positions
    nodes, positions = [], []
    for n in dpg.get_selected_nodes(win.node_editor):
        nodes.append(dpg.get_item_user_data(n))
        positions.append(dpg.get_item_pos(n))

    # serialize the nodes
    for n in nodes:
        n.serialize()

    # wait for all nodes to respond, i.e. their serialized_state is not None
    start = time.time()
    serialized_nodes = []
    for node, pos in zip(nodes, positions):
        while node.serialized_state is None and time.time() - start < timeout:
            # wait for the node to respond or for the timeout to be reached
            time.sleep(0.01)

        # check if we got a response in time
        if node.serialized_state is None:
            # TODO: add proper logging
            print(f"Node {node} did not respond to serialize request.")
            return

        # store the serialized state
        ser = deepcopy(node.serialized_state)
        del ser["out_conns"]
        ser["pos"] = pos
        serialized_nodes.append(ser)

    # store the serialized nodes in the clipboard
    win.node_clipboard = serialized_nodes


def paste_nodes(win):
    """Paste the nodes from the clipboard."""
    if not dpg.is_key_down(dpg.mvKey_Control) or win.node_clipboard is None:
        return

    # add the nodes to the manager
    for node in win.node_clipboard:
        # move node slightly to the bottom right
        pos = node["pos"]
        pos[0] += 30
        pos[1] += 30
        # add the node
        win.manager.add_node(node["_type"], node["category"], params=node["params"], pos=pos)


# the key handler map maps key press events to functions that handle them
KEY_HANDLER_MAP = {
    dpg.mvKey_Delete: delete_selected_item,
    # dpg.mvKey_X: delete_selected_item, # TODO: filter out when editing some text field
    dpg.mvKey_Tab: create_node,
    dpg.mvKey_Escape: escape,
    dpg.mvKey_S: save_manager,
    dpg.mvKey_C: copy_selected_nodes,
    dpg.mvKey_V: paste_nodes,
}
