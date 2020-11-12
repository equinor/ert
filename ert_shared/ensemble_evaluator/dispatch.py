from collections import defaultdict


class Dispatcher:
    def __init__(self):
        self.__LOOKUP_MAP = defaultdict(list)

    def register_event_handler(self, event_types):
        def decorator(function):
            nonlocal event_types
            if not isinstance(event_types, set):
                event_types = set({event_types})
            for event_type in event_types:
                self.__LOOKUP_MAP[event_type].append(function)

            def wrapper(*args, **kwargs):
                return function(*args, **kwargs)

            return wrapper

        return decorator

    async def handle_event(self, instance, event):
        for f in self.__LOOKUP_MAP[event["type"]]:
            await f(instance, event)
