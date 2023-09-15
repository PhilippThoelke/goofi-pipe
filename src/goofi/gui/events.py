import dearpygui.dearpygui as dpg


def delete_item(win):
    for node in dpg.get_selected_nodes(win.node_editor):
        win._remove_node(node)
    for link in dpg.get_selected_links(win.node_editor):
        win._remove_link(link)


KEY_MAP = {
    dpg.mvKey_Delete: delete_item,
    dpg.mvKey_X: delete_item,
}
