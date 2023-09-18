import dearpygui.dearpygui as dpg


def click_callback(_, data, win):
    selected = dpg.get_selected_nodes(win.node_editor)
    if len(selected) == 1:
        win._select_node(selected[0])
    else:
        win._select_node(None)


def key_press_callback(_, data, win):
    if data in KEY_MAP:
        KEY_MAP[data](win)


def delete_item(win):
    for node in dpg.get_selected_nodes(win.node_editor):
        win._remove_node(node)
    for link in dpg.get_selected_links(win.node_editor):
        win._remove_link(link)


KEY_MAP = {
    dpg.mvKey_Delete: delete_item,
    dpg.mvKey_X: delete_item,
}
