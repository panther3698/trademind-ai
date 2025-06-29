import time
import threading

class CacheManager:
    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()
        self.metrics = {'hits': 0, 'misses': 0}
        self.ttls = {
            'market': 5,
            'news': 60,
            'regime': 300
        }

    def get(self, key, category):
        with self.lock:
            entry = self.cache.get((key, category))
            if entry:
                value, timestamp = entry
                if time.time() - timestamp < self.ttls.get(category, 60):
                    self.metrics['hits'] += 1
                    return value
                else:
                    del self.cache[(key, category)]
            self.metrics['misses'] += 1
            return None

    def set(self, key, value, category):
        with self.lock:
            self.cache[(key, category)] = (value, time.time())

    def invalidate(self, key, category):
        with self.lock:
            if (key, category) in self.cache:
                del self.cache[(key, category)]

    def get_metrics(self):
        with self.lock:
            return dict(self.metrics) 