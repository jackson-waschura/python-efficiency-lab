"""
This test suite evaluates the correctness of the graph implementation in graph.py.
We test the following edge cases and scenarios for both Graph and WeightedGraph:

1. Creating an empty graph (directed and undirected)
2. Adding nodes
3. Adding edges
4. Removing edges
5. Removing nodes
6. Getting neighbors
7. String representation
8. Edge cases:
   - Adding duplicate nodes/edges
   - Removing non-existent nodes/edges
   - Getting neighbors of non-existent nodes
9. For WeightedGraph:
   - Adding weighted edges
   - Getting edge weights
   - Updating edge weights
"""

import unittest
from pel.ds.graph import Graph, WeightedGraph

class TestGraph(unittest.TestCase):
    def test_create_empty_graph(self):
        """
        Test creating empty directed and undirected graphs.
        """
        undirected_graph = Graph()
        directed_graph = Graph(directed=True)
        
        self.assertFalse(undirected_graph.directed)
        self.assertTrue(directed_graph.directed)
        self.assertEqual(len(undirected_graph.nodes), 0)
        self.assertEqual(len(directed_graph.nodes), 0)

    def test_add_nodes(self):
        """
        Test adding nodes to the graph.
        """
        graph = Graph()
        graph.add_node(1)
        graph.add_node(2)
        graph.add_node(3)
        
        self.assertEqual(len(graph.nodes), 3)
        self.assertIn(1, graph.nodes)
        self.assertIn(2, graph.nodes)
        self.assertIn(3, graph.nodes)

    def test_add_edges(self):
        """
        Test adding edges to the graph.
        """
        graph = Graph()
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        
        self.assertIn(2, graph.nodes[1])
        self.assertIn(1, graph.nodes[2])
        self.assertIn(3, graph.nodes[2])
        self.assertIn(2, graph.nodes[3])

    def test_add_edges_directed(self):
        """
        Test adding edges to a directed graph.
        """
        graph = Graph(directed=True)
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        
        self.assertIn(2, graph.nodes[1])
        self.assertNotIn(1, graph.nodes[2])
        self.assertIn(3, graph.nodes[2])
        self.assertNotIn(2, graph.nodes[3])

    def test_remove_edge(self):
        """
        Test removing an edge from the graph.
        """
        graph = Graph()
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.remove_edge(1, 2)
        
        self.assertNotIn(2, graph.nodes[1])
        self.assertNotIn(1, graph.nodes[2])
        self.assertIn(3, graph.nodes[2])

    def test_remove_node(self):
        """
        Test removing a node from the graph.
        """
        graph = Graph()
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.remove_node(2)
        
        self.assertNotIn(2, graph.nodes)
        self.assertNotIn(2, graph.nodes[1])
        self.assertNotIn(2, graph.nodes[3])

    def test_get_neighbors(self):
        """
        Test getting neighbors of a node.
        """
        graph = Graph()
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        
        neighbors = graph.get_neighbors(1)
        self.assertEqual(len(neighbors), 2)
        self.assertIn(2, neighbors)
        self.assertIn(3, neighbors)

    def test_string_representation(self):
        """
        Test the string representation of a more complex graph.
        """
        graph = Graph()
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(2, 4)
        graph.add_edge(3, 4)
        graph.add_node(5)  # Isolated node
        
        expected_str = "Undirected graph with nodes: [1, 2, 3, 4, 5]"
        self.assertEqual(str(graph), expected_str)

    def test_add_duplicate_node(self):
        """
        Test adding a duplicate node to the graph.
        """
        graph = Graph()
        graph.add_node(1)
        graph.add_node(1)
        
        self.assertEqual(len(graph.nodes), 1)

    def test_add_duplicate_edge(self):
        """
        Test adding a duplicate edge to the graph.
        """
        graph = Graph()
        graph.add_edge(1, 2)
        graph.add_edge(1, 2)
        
        self.assertEqual(len(graph.nodes[1]), 1)
        self.assertEqual(len(graph.nodes[2]), 1)

    def test_remove_nonexistent_edge(self):
        """
        Test removing a non-existent edge from the graph.
        """
        graph = Graph()
        graph.add_node(1)
        graph.add_node(2)
        graph.remove_edge(1, 2)
        
        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(len(graph.nodes[1]), 0)
        self.assertEqual(len(graph.nodes[2]), 0)

    def test_remove_nonexistent_node(self):
        """
        Test removing a non-existent node from the graph.
        """
        graph = Graph()
        graph.add_node(1)
        graph.remove_node(2)
        
        self.assertEqual(len(graph.nodes), 1)
        self.assertIn(1, graph.nodes)

    def test_get_neighbors_nonexistent_node(self):
        """
        Test getting neighbors of a non-existent node.
        """
        graph = Graph()
        neighbors = graph.get_neighbors(1)
        
        self.assertEqual(len(neighbors), 0)

class TestWeightedGraph(unittest.TestCase):
    def test_create_empty_weighted_graph(self):
        """
        Test creating empty directed and undirected weighted graphs.
        """
        undirected_graph = WeightedGraph()
        directed_graph = WeightedGraph(directed=True)
        
        self.assertFalse(undirected_graph.directed)
        self.assertTrue(directed_graph.directed)
        self.assertEqual(len(undirected_graph.nodes), 0)
        self.assertEqual(len(directed_graph.nodes), 0)

    def test_add_weighted_edges(self):
        """
        Test adding weighted edges to the graph.
        """
        graph = WeightedGraph()
        graph.add_edge(1, 2, 0.5)
        graph.add_edge(2, 3, 1.5)
        
        self.assertEqual(graph.nodes[1][2], 0.5)
        self.assertEqual(graph.nodes[2][1], 0.5)
        self.assertEqual(graph.nodes[2][3], 1.5)
        self.assertEqual(graph.nodes[3][2], 1.5)

    def test_add_weighted_edges_directed(self):
        """
        Test adding weighted edges to a directed graph.
        """
        graph = WeightedGraph(directed=True)
        graph.add_edge(1, 2, 0.5)
        graph.add_edge(2, 3, 1.5)
        
        self.assertEqual(graph.nodes[1][2], 0.5)
        self.assertNotIn(1, graph.nodes[2])
        self.assertEqual(graph.nodes[2][3], 1.5)
        self.assertNotIn(2, graph.nodes[3])

    def test_get_weight(self):
        """
        Test getting the weight of an edge.
        """
        graph = WeightedGraph()
        graph.add_edge(1, 2, 0.5)
        
        self.assertEqual(graph.get_weight(1, 2), 0.5)
        self.assertEqual(graph.get_weight(2, 1), 0.5)

    def test_update_weight(self):
        """
        Test updating the weight of an edge.
        """
        graph = WeightedGraph()
        graph.add_edge(1, 2, 0.5)
        graph.add_edge(1, 2, 1.0)
        
        self.assertEqual(graph.get_weight(1, 2), 1.0)
        self.assertEqual(graph.get_weight(2, 1), 1.0)

    def test_remove_weighted_edge(self):
        """
        Test removing a weighted edge from the graph.
        """
        graph = WeightedGraph()
        graph.add_edge(1, 2, 0.5)
        graph.add_edge(2, 3, 1.5)
        graph.remove_edge(1, 2)
        
        self.assertNotIn(2, graph.nodes[1])
        self.assertNotIn(1, graph.nodes[2])
        self.assertIn(3, graph.nodes[2])

    def test_remove_node_weighted(self):
        """
        Test removing a node from the weighted graph.
        """
        graph = WeightedGraph()
        graph.add_edge(1, 2, 0.5)
        graph.add_edge(2, 3, 1.5)
        graph.remove_node(2)
        
        self.assertNotIn(2, graph.nodes)
        self.assertNotIn(2, graph.nodes[1])
        self.assertNotIn(2, graph.nodes[3])

    def test_get_neighbors_weighted(self):
        """
        Test getting neighbors of a node in a weighted graph.
        """
        graph = WeightedGraph()
        graph.add_edge(1, 2, 0.5)
        graph.add_edge(1, 3, 1.5)
        
        neighbors = graph.get_neighbors(1)
        self.assertEqual(len(neighbors), 2)
        self.assertEqual(neighbors[2], 0.5)
        self.assertEqual(neighbors[3], 1.5)

    def test_string_representation_weighted(self):
        """
        Test the string representation of a more complex weighted graph.
        """
        graph = WeightedGraph()
        graph.add_edge(1, 2, 0.5)
        graph.add_edge(1, 3, 1.5)
        graph.add_edge(2, 4, 2.0)
        graph.add_edge(3, 4, 3.5)
        graph.add_node(5)  # Isolated node
        
        expected_str = "Undirected weighted graph with nodes: [1, 2, 3, 4, 5]"
        self.assertEqual(str(graph), expected_str)

    def test_get_weight_nonexistent_edge(self):
        """
        Test getting the weight of a non-existent edge.
        """
        graph = WeightedGraph()
        graph.add_node(1)
        graph.add_node(2)
        
        with self.assertRaises(KeyError):
            graph.get_weight(1, 2)

if __name__ == '__main__':
    unittest.main()
