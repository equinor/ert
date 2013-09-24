class Observable(object):

    def __init__(self, name):
        self.observers = {}
        self.name = name

    def addEvent(self, event):
        if not event in self.observers:
            self.observers[event] = []

    def __contains__(self, event):
        return event in self.observers

    def attach(self, event, observer_function):
        assert callable(observer_function), "The observer function must be callable!"

        if not event in self:
            raise LookupError("Observer do not have an event of type: %s" % str(event))

        if not observer_function in self.observers[event]:
            self.observers[event].append(observer_function)

    def detach(self, event, observer_function):

        if not event in self:
            raise LookupError("Observer do not have an event of type: %s" % str(event))

        if not observer_function in self.observers[event]:
            raise ValueError("Observer is not observing event '%s' from observable with name: %s" % (str(event), self.name))

        self.observers[event].remove(observer_function)

    def notify(self, event, debug_message=None):
        if not event in self:
            raise LookupError("Observer '%s' do not have an event of type: %s" % (self.name, str(event)))

        # if len(self.observers[event]) == 0:
        #     raise ValueError("Observer has no observers for event of type: %s " % event)

        if debug_message is not None:
            print("Notification: %s - %s " % (event, str(debug_message)))

        for observer in self.observers[event]:
            observer()

    def __str__(self):
        return "Observable %s: %s" % (self.name, str(self.observers.keys()))



