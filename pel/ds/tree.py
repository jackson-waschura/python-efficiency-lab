"""
A tree is a data structure that stores items in a hierarchy.

Each item in the tree is a node, which contains a value and references to its children.

In Python, trees are typically implemented using a recursive data type.
"""

from typing import Any

class TreeNode:
    """
    A node in a tree.
    """

    def __init__(self, value: Any) -> None:
        """
        Initialize the node with a given value.
        """
        self.value = value
        self.children = []

    def add_child(self, child: 'TreeNode') -> None:
        """
        Add a child to the node.
        """
        self.children.append(child)

class BinaryTreeNode:
    """
    A node in a binary tree.
    """

    def __init__(self, value: Any, left: 'BinaryTreeNode' = None, right: 'BinaryTreeNode' = None) -> None:
        self.value = value
        self.left = left
        self.right = right

    @classmethod
    def from_list(cls, values: list[Any]) -> 'BinaryTreeNode':
        """
        Create a binary tree from a list of values.

        The list is assumed to be a level-order traversal of the tree.
        None values represent empty nodes.

        Args:
            values (list[Any]): List of values in level-order traversal.

        Returns:
            BinaryTreeNode: Root node of the constructed binary tree.
        """
        if not values or values[0] is None:
            return None

        # Create the root node
        root = cls(values[0])
        
        # Use a queue to keep track of nodes at each level
        queue = [root]
        i = 1  # Index to track current position in the values list

        while queue and i < len(values):
            current_node = queue.pop(0)

            # Left child
            if i < len(values):
                if values[i] is not None:
                    current_node.left = cls(values[i])
                    queue.append(current_node.left)
                i += 1

            # Right child
            if i < len(values):
                if values[i] is not None:
                    current_node.right = cls(values[i])
                    queue.append(current_node.right)
                i += 1

        return root
        
