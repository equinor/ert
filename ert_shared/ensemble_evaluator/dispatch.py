from collections import defaultdict

__LOOKUP_MAP = {}

__group_id_counter = 0

def dispatch_handler(cls):
    global __group_id_counter
__  __LOOKUP_MAP[__group_id_counter] = defaultdict(list)

    class Wrapper(object):

        def __init__(self, *args, **kwargs):
            self.wrapped = cls(*args, **kwargs)
            self.wrapped.__group_id = __group_id_counter

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    __group_id_counter += 1

    return Wrapper

def register_event_handler(event_types):
    def decorator(function):
        for event_type in event_types:
            __LOOKUP_MAP[event_type].append(function)

        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)

        return wrapper

    return decorator


async def handle_event(self, event):
    for f in __LOOKUP_MAP[event["type"]]:
        await f(self, event)
