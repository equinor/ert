import os


def autoCompleteList(text, items):
    if not text:
        completions = items
    else:
        completions = [item for item in items if item.lower().startswith(text.lower())]
    return completions


def createParameterizedHelpFunction(parameters, help_message):
    def helpFunction(self):
        return parameters, help_message

    return helpFunction


def assertConfigLoaded(func):
    def wrapper(self, *args, **kwargs):
        # prefixes should be either do_ or complete_
        if func.__name__.startswith("complete_"):
            result = []
            verbose = False
        else:
            result = False
            verbose = True

        if self.isConfigLoaded(verbose=verbose):
            result = func(self, *args, **kwargs)

        return result

    wrapper.__doc__ = func.__doc__
    wrapper.__name__ = func.__name__

    return wrapper


def pathify(head, tail):
    path = os.path.join(head, tail)
    if os.path.isdir(path):
        return "%s/" % tail
    return tail


def getPossibleFilenameCompletions(text):
    head, tail = os.path.split(text.strip())
    if head == "":  # no head
        head = "."
    files = os.listdir(head)
    return [pathify(head, f) for f in files if f.startswith(tail)]


def extractFullArgument(line, endidx):
    newstart = line.rfind(" ", 0, endidx)
    return line[newstart:endidx].strip()