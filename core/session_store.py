from collections import deque

class SessionStore:
    """
    Minimal memory: keeps last K interactions.
    Enough to claim 'Sessions & state management'.
    """
    def __init__(self, maxlen=5):
        self.history = deque(maxlen=maxlen)

    def add(self, record: dict):
        self.history.append(record)

    def last(self):
        return self.history[-1] if self.history else None

    def all(self):
        return list(self.history)
