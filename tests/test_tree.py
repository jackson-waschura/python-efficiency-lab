import unittest
from pel.ds import TreeNode, BinaryTreeNode

"""
This test suite evaluates the correctness of the tree implementation in tree.py.
We test the following edge cases and scenarios:

1. Creating and adding children to a TreeNode
2. Creating a BinaryTreeNode with left and right children
3. Creating a BinaryTree from a list of values (BinaryTreeNode.from_list method)
   - Empty list
   - List with only root
   - List with root and left child
   - List with root and right child
   - List with multiple levels and None values
   - List with all None values except root
4. Verifying the structure of the created binary tree
"""

class TestTreeNode(unittest.TestCase):
    def test_create_and_add_children(self):
        """
        Test creating a TreeNode and adding children. This ensures the basic functionality of the TreeNode class works correctly.
        """
        root = TreeNode(1)
        child1 = TreeNode(2)
        child2 = TreeNode(3)
        
        root.add_child(child1)
        root.add_child(child2)
        
        self.assertEqual(root.value, 1)
        self.assertEqual(len(root.children), 2)
        self.assertEqual(root.children[0].value, 2)
        self.assertEqual(root.children[1].value, 3)

class TestBinaryTreeNode(unittest.TestCase):
    def test_create_with_children(self):
        """
        Test creating a BinaryTreeNode with left and right children. This verifies the basic structure of a binary tree node.
        """
        left = BinaryTreeNode(2)
        right = BinaryTreeNode(3)
        root = BinaryTreeNode(1, left, right)
        
        self.assertEqual(root.value, 1)
        self.assertEqual(root.left.value, 2)
        self.assertEqual(root.right.value, 3)

    def test_from_list_empty(self):
        """
        Test creating a binary tree from an empty list. This checks the edge case of an empty input.
        """
        tree = BinaryTreeNode.from_list([])
        self.assertIsNone(tree)

    def test_from_list_root_only(self):
        """
        Test creating a binary tree with only a root node. This verifies the minimal case of a binary tree.
        """
        tree = BinaryTreeNode.from_list([1])
        self.assertEqual(tree.value, 1)
        self.assertIsNone(tree.left)
        self.assertIsNone(tree.right)

    def test_from_list_root_and_left(self):
        """
        Test creating a binary tree with a root and left child. This checks the correct assignment of a left child.
        """
        tree = BinaryTreeNode.from_list([1, 2])
        self.assertEqual(tree.value, 1)
        self.assertEqual(tree.left.value, 2)
        self.assertIsNone(tree.right)

    def test_from_list_root_and_right(self):
        """
        Test creating a binary tree with a root and right child. This verifies the correct handling of a None left child and assignment of a right child.
        """
        tree = BinaryTreeNode.from_list([1, None, 3])
        self.assertEqual(tree.value, 1)
        self.assertIsNone(tree.left)
        self.assertEqual(tree.right.value, 3)

    def test_from_list_multiple_levels(self):
        """
        Test creating a binary tree with multiple levels. This ensures the correct structure of a more complex tree.
        """
        tree = BinaryTreeNode.from_list([1, 2, 3, 4, None, 6, 7])
        self.assertEqual(tree.value, 1)
        self.assertEqual(tree.left.value, 2)
        self.assertEqual(tree.right.value, 3)
        self.assertEqual(tree.left.left.value, 4)
        self.assertIsNone(tree.left.right)
        self.assertEqual(tree.right.left.value, 6)
        self.assertEqual(tree.right.right.value, 7)

    def test_from_list_with_nones(self):
        """
        Test creating a binary tree with None values in the input list. This checks the correct handling of None values in non-leaf positions.
        """
        tree = BinaryTreeNode.from_list([1, None, None, None, 5])
        self.assertEqual(tree.value, 1)
        self.assertIsNone(tree.left)
        self.assertIsNone(tree.right)

    def test_from_list_all_nones_except_root(self):
        """
        Test creating a binary tree with all None values except the root. This verifies the correct handling of a tree with only a root node and all other positions as None.
        """
        tree = BinaryTreeNode.from_list([1, None, None, None, None, None, None])
        self.assertEqual(tree.value, 1)
        self.assertIsNone(tree.left)
        self.assertIsNone(tree.right)

if __name__ == '__main__':
    unittest.main()
