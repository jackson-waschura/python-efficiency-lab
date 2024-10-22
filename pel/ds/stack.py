"""
A stack is a linear data structure that follows the Last In First Out (LIFO) principle.

In Python, we can implement a stack using a list. However, for educational purposes,
we'll create a custom Stack class.
"""

from typing import Any, List

class Stack:
    """
    A stack implementation using a list.
    """

    def __init__(self, items: List[Any] | None = None) -> None:
        """
        Initialize the stack with optional initial items.

        Args:
            items (List[Any] | None): Initial items to push onto the stack. Default is None.
        """
        self.items: List[Any] = items if items is not None else []

    def push(self, item: Any) -> None:
        """
        Push an item onto the top of the stack.

        Args:
            item (Any): The item to push onto the stack.
        """
        self.items.append(item)

    def pop(self) -> Any:
        """
        Remove and return the top item from the stack.

        Returns:
            Any: The top item from the stack.

        Raises:
            IndexError: If the stack is empty.
        """
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items.pop()

    def peek(self) -> Any:
        """
        Return the top item from the stack without removing it.

        Returns:
            Any: The top item from the stack.

        Raises:
            IndexError: If the stack is empty.
        """
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items[-1]

    def is_empty(self) -> bool:
        """
        Check if the stack is empty.

        Returns:
            bool: True if the stack is empty, False otherwise.
        """
        return len(self.items) == 0

    def __len__(self) -> int:
        """
        Return the number of items in the stack.

        Returns:
            int: The number of items in the stack.
        """
        return len(self.items)

    def __str__(self) -> str:
        """
        Return a string representation of the stack.

        Returns:
            str: A string representation of the stack.
        """
        return f"Stack: {self.items}"
