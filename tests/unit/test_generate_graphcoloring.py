import pytest

from pydcop.commands.generators.graphcoloring import generate_grid_graph, \
    generate_scalefree_graph


def test_grid_graph_raises_with_invalide_size():
    with pytest.raises(ValueError):
        generate_grid_graph(5)


def test_grid_graph():
    graph = generate_grid_graph(16)
    assert len(list(graph.nodes)) == 16
    assert len(list(graph.edges)) == 24


def test_generate_scale_free():
    graph = generate_scalefree_graph(10, 2, False)
    assert len(graph.nodes) == 10