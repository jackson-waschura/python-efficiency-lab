"""
A heap is a data structure that stores items so that the smallest (or largest) item is always at the top.

Heaps are commonly implemented using arrays, but they can also be implemented using trees.
This implementation uses an array-based approach for efficiency.

NOTE: While the following example is helpful for understanding what a heap is, it is definitely not
the best way to implement a heap. The easiest way is to use the heapq module.
Here's an example of how to use it:

# heapq methods are in-place, so the original list is modified.
import heapq

heap = []
heapq.heappush(heap, item)
min_val = heapq.heappop(heap)

# Construct a heap from a list of elements in O(n) time.
my_inputs = [...]
heapq.heapify(my_inputs)

# To use heapq effectively to implement a priority queue, you can use tuples with priority values
# as the first element of the tuple like this:
my_priority_queue = []
heapq.heappush(my_priority_queue, (priority_val, item))
item = heapq.heappop(my_priority_queue)[1]
"""

from typing import Any, List, Callable

class Heap:
    """
    A heap data structure that can be either a min heap or a max heap.
    """

    def __init__(self, is_min_heap: bool = True) -> None:
        """
        Initialize the heap.

        Args:
            is_min_heap (bool): If True, create a min heap. If False, create a max heap. Default is True.
        """
        self.heap: List[Any] = []
        self.is_min_heap = is_min_heap
        self.compare: Callable[[Any, Any], bool] = (lambda x, y: x < y) if is_min_heap else (lambda x, y: x > y)

    @classmethod
    def from_list(cls, elements: List[Any], is_min_heap: bool = True) -> 'Heap':
        """
        Construct a heap from a list of elements in O(n) time.

        Args:
            elements (List[Any]): The list of elements to construct the heap from.
            is_min_heap (bool): If True, create a min heap. If False, create a max heap. Default is True.

        Returns:
            Heap: A new Heap instance containing the given elements.
        """
        heap = cls(is_min_heap)
        heap.heap = elements.copy()
        for i in range(len(heap.heap) // 2 - 1, -1, -1):
            heap._sift_down(i)
        return heap

    def parent(self, i: int) -> int:
        """
        Get the parent index of a given index.

        Args:
            i (int): The index of the current element.

        Returns:
            int: The index of the parent element.
        """
        return (i - 1) // 2

    def left_child(self, i: int) -> int:
        """
        Get the left child index of a given index.

        Args:
            i (int): The index of the current element.

        Returns:
            int: The index of the left child element.
        """
        return 2 * i + 1

    def right_child(self, i: int) -> int:
        """
        Get the right child index of a given index.

        Args:
            i (int): The index of the current element.

        Returns:
            int: The index of the right child element.
        """
        return 2 * i + 2

    def swap(self, i: int, j: int) -> None:
        """
        Swap two elements in the heap.

        Args:
            i (int): The index of the first element.
            j (int): The index of the second element.
        """
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def push(self, item: Any) -> None:
        """
        Add an item to the heap.

        Args:
            item (Any): The item to add to the heap.
        """
        self.heap.append(item)
        self._sift_up(len(self.heap) - 1)

    def pop(self) -> Any:
        """
        Remove and return the top item from the heap.

        Returns:
            Any: The top item from the heap.

        Raises:
            IndexError: If the heap is empty.
        """
        if not self.heap:
            raise IndexError("Heap is empty")
        if len(self.heap) == 1:
            return self.heap.pop()
        top_item = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._sift_down(0)
        return top_item

    def peek(self) -> Any:
        """
        Return the top item from the heap without removing it.

        Returns:
            Any: The top item from the heap.

        Raises:
            IndexError: If the heap is empty.
        """
        if not self.heap:
            raise IndexError("Heap is empty")
        return self.heap[0]

    def _sift_up(self, i: int) -> None:
        """
        Sift up the item at the given index to maintain the heap property.

        Args:
            i (int): The index of the item to sift up.
        """
        while i > 0 and self.compare(self.heap[i], self.heap[self.parent(i)]):
            self.swap(i, self.parent(i))
            i = self.parent(i)

    def _sift_down(self, i: int) -> None:
        """
        Sift down the item at the given index to maintain the heap property.

        Args:
            i (int): The index of the item to sift down.
        """
        while True:
            min_index = i
            left = self.left_child(i)
            right = self.right_child(i)

            if left < len(self.heap) and self.compare(self.heap[left], self.heap[min_index]):
                min_index = left
            if right < len(self.heap) and self.compare(self.heap[right], self.heap[min_index]):
                min_index = right

            if i == min_index:
                break

            self.swap(i, min_index)
            i = min_index

    def __len__(self) -> int:
        """
        Return the number of items in the heap.

        Returns:
            int: The number of items in the heap.
        """
        return len(self.heap)

    def __str__(self) -> str:
        """
        Return a string representation of the heap.

        Returns:
            str: A string representation of the heap.
        """
        return f"{'Min' if self.is_min_heap else 'Max'} Heap: {self.heap}"
