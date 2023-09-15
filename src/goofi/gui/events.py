import dearpygui.dearpygui as dpg


def delete(win):
    for node in dpg.get_selected_nodes(win.node_editor):
        win._remove_node(node)
    for link in dpg.get_selected_links(win.node_editor):
        win._remove_link(link)


KEY_MAP = {
    261: delete,  # del
    88: delete,  # x
}
