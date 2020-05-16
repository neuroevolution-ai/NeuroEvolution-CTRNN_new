
def walk_dict(node, callback_node, depth=0):
    for key, item in node.items():
        if isinstance(item,dict):
            callback_node(key, item, depth, False)
            walk_dict(item, callback_node, depth + 1)
        else:
            callback_node(key, item, depth, True)
