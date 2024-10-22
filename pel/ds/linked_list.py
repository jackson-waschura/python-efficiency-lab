"""
A Linked List is a linear data structure that consists of a sequence of nodes,
where each node contains a value and a reference (or pointer) to the next node in the sequence.

A Doubly Linked List is a linked list where each node has a reference to both the next and previous node.
"""

from typing import Any, Iterator

class Node:
    """
    A node in a singly linked list.
    """

    def __init__(self, value: Any, next_node: 'Node | None' = None) -> None:
        self.value = value
        self.next = next_node

class LinkedList:
    """
    A singly linked list implementation.
    """

    def __init__(self) -> None:
        self.head: Node | None = None
        self.tail: Node | None = None
        self.size = 0

    def append(self, value: Any) -> None:
        """
        Add a new node with the given value to the end of the list.
        """
        new_node = Node(value)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1

    def prepend(self, value: Any) -> None:
        """
        Add a new node with the given value to the beginning of the list.
        """
        new_node = Node(value, self.head)
        self.head = new_node
        if self.tail is None:
            self.tail = new_node
        self.size += 1

    def delete(self, value: Any) -> bool:
        """
        Delete the first occurrence of a node with the given value.
        """
        if self.head is None:
            return False

        if self.head.value == value:
            self.head = self.head.next
            self.size -= 1
            if self.head is None:
                self.tail = None
            return True

        current = self.head
        while current.next:
            if current.next.value == value:
                current.next = current.next.next
                self.size -= 1
                if current.next is None:
                    self.tail = current
                return True
            current = current.next

        return False

    def __len__(self) -> int:
        return self.size

    def __str__(self) -> str:
        values = []
        current = self.head
        while current:
            values.append(str(current.value))
            current = current.next
        return " -> ".join(values)

    def __iter__(self) -> Iterator[Any]:
        """
        Return an iterator for the linked list.
        """
        current = self.head
        while current:
            yield current.value
            current = current.next

class DoublyNode:
    """
    A node in a doubly linked list.
    """

    def __init__(self, value: Any, prev_node: 'DoublyNode | None' = None, next_node: 'DoublyNode | None' = None) -> None:
        self.value = value
        self.prev = prev_node
        self.next = next_node

class DoublyLinkedList:
    """
    A doubly linked list implementation.
    """

    def __init__(self) -> None:
        self.head: DoublyNode | None = None
        self.tail: DoublyNode | None = None
        self.size = 0

    def append(self, value: Any) -> None:
        """
        Add a new node with the given value to the end of the list.
        """
        new_node = DoublyNode(value)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1

    def prepend(self, value: Any) -> None:
        """
        Add a new node with the given value to the beginning of the list.
        """
        new_node = DoublyNode(value, next_node=self.head)
        if self.head:
            self.head.prev = new_node
        self.head = new_node
        if self.tail is None:
            self.tail = new_node
        self.size += 1

    def delete(self, value: Any) -> bool:
        """
        Delete the first occurrence of a node with the given value.
        """
        current = self.head
        while current:
            if current.value == value:
                if current.prev:
                    current.prev.next = current.next
                else:
                    self.head = current.next

                if current.next:
                    current.next.prev = current.prev
                else:
                    self.tail = current.prev

                self.size -= 1
                return True
            current = current.next

        return False

    def __len__(self) -> int:
        return self.size

    def __str__(self) -> str:
        values = []
        current = self.head
        while current:
            values.append(str(current.value))
            current = current.next
        return " <-> ".join(values)

    def __iter__(self, reverse: bool = False) -> Iterator[Any]:
        """
        Return an iterator for the doubly linked list.
        """
        current = self.tail if reverse else self.head
        while current:
            yield current.value
            current = current.prev if reverse else current.next
