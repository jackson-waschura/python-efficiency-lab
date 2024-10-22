"""
A graph is a data structure that stores items in a network.

Each item in the graph is a node, which contains a value and references to its neighbors.
This implementation provides both directed and undirected graph functionality.
"""

from typing import Any, Dict, Set

class Graph:
    """
    An unweighted graph implementation using an adjacency list.
    """

    def __init__(self, directed: bool = False) -> None:
        """
        Initialize the graph.

        Args:
            directed (bool): If True, the graph is directed. Default is False.
        """
        self.directed = directed
        self.nodes: Dict[Any, Set[Any]] = {}

    def add_node(self, node: Any) -> None:
        """
        Add a node to the graph if it doesn't already exist.

        Args:
            node (Any): The node to add.
        """
        if node not in self.nodes:
            self.nodes[node] = set()

    def add_edge(self, start: Any, end: Any) -> None:
        """
        Add an edge between two nodes.

        Args:
            start (Any): The starting node.
            end (Any): The ending node.
        """
        self.add_node(start)
        self.add_node(end)
        self.nodes[start].add(end)
        if not self.directed:
            self.nodes[end].add(start)

    def remove_edge(self, start: Any, end: Any) -> None:
        """
        Remove an edge between two nodes.

        Args:
            start (Any): The starting node.
            end (Any): The ending node.
        """
        if start in self.nodes and end in self.nodes[start]:
            self.nodes[start].remove(end)
            if not self.directed and start in self.nodes[end]:
                self.nodes[end].remove(start)

    def remove_node(self, node: Any) -> None:
        """
        Remove a node and all its edges from the graph.

        Args:
            node (Any): The node to remove.
        """
        if node in self.nodes:
            del self.nodes[node]
            # Remove edges pointing to this node in other nodes
            for other_node in self.nodes:
                if node in self.nodes[other_node]:
                    self.nodes[other_node].remove(node)

    def get_neighbors(self, node: Any) -> Set[Any]:
        """
        Get the neighbors of a node.

        Args:
            node (Any): The node to get neighbors for.

        Returns:
            Set[Any]: A set of neighboring nodes.
        """
        return self.nodes.get(node, set())

    def __str__(self) -> str:
        """
        Return a string representation of the graph.

        Returns:
            str: A string representation of the graph.
        """
        return f"{'Directed' if self.directed else 'Undirected'} graph with nodes: {list(self.nodes.keys())}"


class WeightedGraph:
    """
    A weighted graph implementation using an adjacency list.
    """

    def __init__(self, directed: bool = False) -> None:
        """
        Initialize the weighted graph.

        Args:
            directed (bool): If True, the graph is directed. Default is False.
        """
        self.directed = directed
        self.nodes: Dict[Any, Dict[Any, float]] = {}

    def add_node(self, node: Any) -> None:
        """
        Add a node to the graph if it doesn't already exist.

        Args:
            node (Any): The node to add.
        """
        if node not in self.nodes:
            self.nodes[node] = {}

    def add_edge(self, start: Any, end: Any, weight: float) -> None:
        """
        Add a weighted edge between two nodes.

        Args:
            start (Any): The starting node.
            end (Any): The ending node.
            weight (float): The weight of the edge.
        """
        self.add_node(start)
        self.add_node(end)
        self.nodes[start][end] = weight
        if not self.directed:
            self.nodes[end][start] = weight

    def remove_edge(self, start: Any, end: Any) -> None:
        """
        Remove a weighted edge between two nodes.

        Args:
            start (Any): The starting node.
            end (Any): The ending node.
        """
        if start in self.nodes and end in self.nodes[start]:
            del self.nodes[start][end]
            if not self.directed and start in self.nodes[end]:
                del self.nodes[end][start]

    def remove_node(self, node: Any) -> None:
        """
        Remove a node and all its edges from the graph.

        Args:
            node (Any): The node to remove.
        """
        if node in self.nodes:
            del self.nodes[node]
            # Remove edges pointing to this node in other nodes
            for other_node in self.nodes:
                if node in self.nodes[other_node]:
                    del self.nodes[other_node][node]

    def get_weight(self, start: Any, end: Any) -> float:
        """
        Get the weight of an edge between two nodes.

        Args:
            start (Any): The starting node.
            end (Any): The ending node.

        Returns:
            float: The weight of the edge.

        Raises:
            KeyError: If the edge does not exist.
        """
        if start in self.nodes and end in self.nodes[start]:
            return self.nodes[start][end]
        raise KeyError(f"Edge from {start} to {end} does not exist")

    def get_neighbors(self, node: Any) -> Dict[Any, float]:
        """
        Get the neighbors of a node and their edge weights.

        Args:
            node (Any): The node to get neighbors for.

        Returns:
            Dict[Any, float]: A dictionary of neighboring nodes and their edge weights.
        """
        return self.nodes.get(node, {})

    def __str__(self) -> str:
        """
        Return a string representation of the weighted graph.

        Returns:
            str: A string representation of the weighted graph.
        """
        return f"{'Directed' if self.directed else 'Undirected'} weighted graph with nodes: {list(self.nodes.keys())}"
