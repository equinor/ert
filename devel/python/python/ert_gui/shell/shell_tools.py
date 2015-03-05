import os


def autoCompleteList(text, items):
    if not text:
        completions = items
    else:
        completions = [item for item in items if item.lower().startswith(text.lower())]
    return completions


def autoCompleteListWithSeparator(text, items, separator=":"):
    if separator in text:
        auto_complete_list = autoCompleteList(text, items)
        auto_complete_list = [item[item.rfind(":") + 1:] for item in auto_complete_list if separator in item]
    else:
        auto_complete_list = autoCompleteList(text, items)

    return auto_complete_list


def createParameterizedHelpFunction(parameters, help_message):
    def helpFunction(self):
        return parameters, help_message

    return helpFunction


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