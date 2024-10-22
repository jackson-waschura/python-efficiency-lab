"""
A Map is a data structure that stores key-value pairs.

In Python, this is typically realized with a dict. But for educational
purposes, we'll implement our own.

In reality, we would use the built-in dict type, which uses open addressing
with hash-dependent probing to resolve collisions.
"""

from typing import Any

class Map:
    """
    A map implemented as an array of buckets which contain key-value pairs.

    Collisions are resolved by chaining. If the average number of items per bucket
    exceeds 1.5, the number of buckets is doubled and all items are rehashed.
    """

    def __init__(self, num_buckets: int = 32) -> None:
        """
        Initialize the map with a given number of buckets.
        """
        self.num_buckets = num_buckets
        self.buckets = [[] for _ in range(self.num_buckets)]
        self.num_items = 0

    def __len__(self) -> int:
        """
        Return the number of items in the map.
        """
        return self.num_items

    def __getitem__(self, key: Any) -> Any:
        """
        Return the value associated with the given key.
        """
        bucket_index = hash(key) % self.num_buckets
        bucket = self.buckets[bucket_index]
        for k, v in bucket:
            if k == key:
                return v
        raise KeyError(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        """
        Set the value associated with the given key.
        """
        bucket_index = hash(key) % self.num_buckets
        bucket = self.buckets[bucket_index]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self.num_items += 1
        if self.num_items / self.num_buckets > 1.5:
            self.rehash()

    def __delitem__(self, key: Any) -> None:
        """
        Delete the value associated with the given key.
        """
        bucket_index = hash(key) % self.num_buckets
        bucket = self.buckets[bucket_index]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                self.num_items -= 1
                return
        raise KeyError(key)
    
    def rehash(self, new_num_buckets: int | None = None) -> None:
        """
        Rehash the map by doubling the number of buckets and rehashing all items.
        """
        if new_num_buckets is None:
            new_num_buckets = self.num_buckets * 2
        self.num_buckets = new_num_buckets
        new_buckets = [[] for _ in range(self.num_buckets)]
        for bucket in self.buckets:
            for k, v in bucket:
                new_bucket_index = hash(k) % self.num_buckets
                new_buckets[new_bucket_index].append((k, v))
        self.buckets = new_buckets
