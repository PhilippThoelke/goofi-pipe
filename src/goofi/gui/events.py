import dearpygui.dearpygui as dpg

from goofi.node_helpers import list_nodes

################################
######### Mouse Events #########
################################


def click_callback(_, btn, win):
    """Update node selection after a click event."""
    selected = dpg.get_selected_nodes(win.node_editor)
    if len(selected) == 1:
        win._select_node(selected[0])
    else:
        win._select_node(None)


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
    # clean up the state of the GUI
    escape(win)
    dpg.clear_selected_nodes(win.node_editor)
    dpg.clear_selected_links(win.node_editor)
    # create the node inside the manager, which will notify the window
    win.manager.add_node(node.__name__, node.category())


def create_node(win):
    """Opens the window for creating a node. If the window is already open, closes it."""
    if win.create_node_window is not None:
        # close the existing window instance
        dpg.delete_item(win.create_node_window)
        win.create_node_window = None

    # create a new window instance
    win.create_node_window = dpg.add_window(
        label="Create Node", pos=dpg.get_mouse_pos(local=False), no_collapse=True, autosize=True
    )

    # create a dictionary of nodes by category
    categories = {}
    for node in list_nodes():
        cat = node.category()
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(node)

    # create a tab bar with a tab for each category
    with dpg.tab_bar(parent=win.create_node_window):
        for cat, nodes in categories.items():
            with dpg.tab(label=cat):
                # create a button for each node in the category
                for node in nodes:
                    dpg.add_button(label=node.__name__, callback=select_node_callback, user_data=(win, node))


def escape(win):
    """
    Provides a way to cancel the creation of a node by closing the node selection window.
    If it was already closed, clears the current node and link selection.
    """
    if win.create_node_window is not None:
        dpg.delete_item(win.create_node_window)
        win.create_node_window = None
    else:
        dpg.clear_selected_nodes(win.node_editor)
        dpg.clear_selected_links(win.node_editor)


# the key handler map maps key press events to functions that handle them
KEY_HANDLER_MAP = {
    dpg.mvKey_Delete: delete_selected_item,
    dpg.mvKey_X: delete_selected_item,
    dpg.mvKey_Tab: create_node,
    dpg.mvKey_Escape: escape,
}
