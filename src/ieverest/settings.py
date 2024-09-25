from contextlib import contextmanager

from qtpy.QtCore import QSettings


@contextmanager
def qsettings_read_array(qsettings, array_name):
    count = qsettings.beginReadArray(array_name)
    yield count
    qsettings.endArray()


@contextmanager
def qsettings_write_array(qsettings, array_name):
    qsettings.beginWriteArray(array_name)
    yield
    qsettings.endArray()


@contextmanager
def qsettings_group(qsettings, group_name):
    qsettings.beginGroup(group_name)
    yield
    qsettings.endGroup()


_RECENT_FILES_KEY = "recent_files"
_MAX_RECENT_FILES_KEY = "max_recent_files"
_PATH_KEY = "path"


def max_recent_files():
    settings = QSettings()
    return settings.value(_MAX_RECENT_FILES_KEY, 20, type=int)


def set_max_recent_files(count):
    settings = QSettings()
    settings.setValue(_MAX_RECENT_FILES_KEY, count)


def recent_files():
    settings = QSettings()
    files = []
    with qsettings_read_array(settings, _RECENT_FILES_KEY) as count:
        for i in range(count):
            settings.setArrayIndex(i)
            files.append(settings.value(_PATH_KEY, type=str))
    return files


def add_recent_file(filepath):
    files = recent_files()
    files.insert(0, filepath)
    # remove duplicates
    seen = set()
    files = [f for f in files if not (f in seen or seen.add(f))]
    settings = QSettings()
    with qsettings_write_array(settings, _RECENT_FILES_KEY):
        for i in range(min(max_recent_files(), len(files))):
            settings.setArrayIndex(i)
            settings.setValue(_PATH_KEY, files[i])
    return files
