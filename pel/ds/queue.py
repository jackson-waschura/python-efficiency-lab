"""
A Queue is a data structure that stores items in a first-in-first-out (FIFO) manner.

A Priority Queue is a data structure that stores items in a first-in-first-out (FIFO) manner,
but the items have a priority value associated with them such that the item with the lowest
priority value is removed first. For items with the same priority, the order remains FIFO.
"""

from typing import Any, List, Tuple
import heapq
import itertools

class Queue:
    """
    A Queue implementation using a list.
    """

    def __init__(self, values: List[Any] | None = None) -> None:
        self.queue: List[Any] = values if values is not None else []

    def enqueue(self, item: Any) -> None:
        self.queue.append(item)

    def dequeue(self) -> Any:
        return self.queue.pop(0)

    def peek(self) -> Any:
        if self.is_empty():
            raise ValueError("Queue is empty")
        return self.queue[0]

    def __len__(self) -> int:
        return len(self.queue)

    def is_empty(self) -> bool:
        return len(self.queue) == 0

class PriorityQueue:
    """
    A Priority Queue implementation using a heap. Items are stored in a list of tuples, where
    the first element of the tuple is the priority value, the second element is a counter to
    maintain FIFO order for items with the same priority, and the third element is the item.
    
    Items with the lowest priority value are removed first. For items with the same priority,
    the order in which they were added is preserved (FIFO).
    """

    def __init__(self, values: List[Tuple[int, Any]] | None = None) -> None:
        self.counter = itertools.count()
        self.queue: List[Tuple[int, int, Any]] = [(p, next(self.counter), v) for p, v in values] if values else []
        heapq.heapify(self.queue)

    def enqueue(self, item: Any, priority: int) -> None:
        count = next(self.counter)
        heapq.heappush(self.queue, (priority, count, item))

    def dequeue(self) -> Any:
        if self.is_empty():
            raise ValueError("Queue is empty")
        return heapq.heappop(self.queue)[2]

    def peek(self) -> Any:
        if self.is_empty():
            raise ValueError("Queue is empty")
        return self.queue[0][2]
    
    def __len__(self) -> int:
        return len(self.queue)

    def is_empty(self) -> bool:
        return len(self.queue) == 0
