

def diff_vars(a, b):
    """
    This is an example of a constraint defined as a python function, in an external file.
    See `graph_coloring1_func.yaml` for an example on how to use such constraint in
    a yaml dcop file.
    """
    if a == b:
        return 1
    else:
        return 0
