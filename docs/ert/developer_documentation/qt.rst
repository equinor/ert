Testing Qt components in pytest
===============================

Ert’s GUI is written in Qt, using a library like PyQt5 and using the
testing framework ``pytest-qt`` with its test-helper class QtBot. There
are several challenges in integrating a Qt GUI in a Python program,
which this document hopes to alleviate.

The Qt event loop
-----------------

As a GUI framework, Qt is optimised for low resource usage. If an
application is doing nothing, it should use minimal CPU. The program
should only activate when the user interacts with, say, a button. For
this to work, Qt needs to have complete control over the process.

Python, like Javascript, is mostly single-threaded. While Javascript in
the browser is explicitly single-threaded, Python supports creating
threads, but the Global Intepreter Lock makes Python non-concurrent. Because
Qt needs to have complete control over the process, Python functions
should be expected to be short and complete quickly. In our code-base,
this is often not the case. This is equivalent to a web browser having
complete control over the GUI, with any Javascript functions needing to
complete quickly lest the browser hang.

No discussion of Qt is complete without a discussion on its
signals-slots system. This concept is not unique to Qt, however. The
HTML attribute ``onClick`` is exactly equivalent to Qt’s
``QPushButton::clicked`` signal. While in the Javascript world these are
called callbacks, in Qt these are signals and slots. In the web browser,
the web-page is a static HTML document until the user clicks on the
button, causing a short Javascript function to be called. In Qt, the GUI
is static until the user clicks on a button, causing the
``QPushButton::clicked`` signal to be emitted, which eventually calls
the slots connected to it. These functions, too, should return quickly.


Unlike a Javascript callback function, the signal-slots system is more
complicated. When a signal is emitted, it is put on the event queue.
Each thread has its own event queue handler, and will call the slotted
function when it gets around to it. This is an asynchronous model
that makes it easy to deal with threads. In Qt, ``QObject``\ s belong to
threads, inheriting its thread's event queue. Within a ``QObject``, all
functions and fields exist on the same thread, and are in this regard
thread-safe. When interaction happens using the signals-slots system,
events are placed on the correct thread's event queue, meaning this too
happens safely.

If our Qt GUI adheres to these principles, we will be able to benefit
greatly.

QDialog
-------

Ert relies on a number of dialog boxes. These are often modal, meaning
the rest of the GUI is disabled while the dialog is open. Modal dialogs
do not necessarily mean that the underlying code is not allowed to
run, but Ert’s reliance of the ``exec_()`` method makes it so.

The ``exec_()`` method for dialogs makes Qt block until the dialog is
closed. For pytest, this means that the testing code stops entirely. To
work around this, our tests use the ``QTimer.singleShot`` pattern. This
adds a function to Qt’s queue to be run at a later time, usually in a
second. Starting this timer before doing ``exec_()`` will let the given
function to continue the test. This has the following problems: the
function is opaque to pytest and if the function does not complete, the
test hangs. The dialog box never closes, thus the ``exec_()`` call never
returns. The exception thrown by the function is lost as well, caught by
Qt’s exception handler.

The alternative and correct way of handling dialogs is using the
``open()`` method. Like ``exec_()`` it will show and focus the window.
Unlike ``exec_()`` it does not block. The code continues. The test is
then responsible to wait for this object to be “exposed”, which can be
done using the ``qtbot.waitExposed`` function.

Instead of the current pattern of:

.. code:: py

   def test_gui(qtbot):
     dialog = MyDialog()

     def handle_dialog():
       qtbot.waitUntil(lambda: isinstance(QApplication.activeWindow(), MyDialog))
       dialog = QApplication.activeWindow()
       assert isinstance(dialog, MyDialog)

       # do something with dialog

       dialog.close()

     QTimer.singleShot(1000, handle_dialog)
     dialog.exec_()

We should write:

.. code:: py

   def test_gui(qtbot):
     dialog = MyDialog()
     dialog.open()
     qtbot.waitExposed(dialog)

      # do something with dialog

     dialog.close()

Notice that the entire test is in pytest. Assertion errors here will
propagate to the qtbot fixture which will be able to deal with errors
appropriately, as well as shut down Qt correctly.

Importantly, this code does not give control to Qt. In production
this is unwanted, but the test code must be controlled by pytest. Note
that all of ``QtBot``\ s ``wait`` functions have a default timeout of five
seconds, meaning that even a failing test will eventually finish.

QMessageBox
-----------

The ``QTimer.singleShot`` pattern is also found dealing with
QMessageBoxes. These message boxes are used around the codebase to
inform the user of various events. Unlike QDialogs, they are not
complicated.

The ``pytest-qt`` documentation advises us to monkeypatch the
QMessageBox static methods to return the desired outcome immediately.
Instead of having the code manually click the “Yes” or “Ok” button, the
QMessageBox should immediately return the desired state.

I propose we mock QMessageBoxes entirely in the whole test-suite. Their
functionality is simple and we do not care about our ability to press
the “Yes” button, as this is something that is already well-tested by
the Qt developers. Instead, the default for all QMessageBox static
methods is to return whatever the default button is, as well as expose a
testing API that can ensure that the contents of the QMessageBox is what
we expected.

We can add a new fixture ``qmsgbox`` which can be used thusly:

.. code:: py

   def test_msg(qtbot, qmsgbox):
     widget = OperationPerformer()

     with qmsgbox.fatal(title="Failure"):
       widget.makeItFail()

In this case, ``qmsgbox.fatal`` is a contextmanager that overrides
``QMessageBox.fatal`` and at the end of the block
``qtbot.waitSignal``\ s for it to be called, asserting that the title of
the QMessageBox is ``"Failure"``. In the code, the ``QMessageBox.fatal``
simply returns ``QMessageBox.Ok`` immediately, allowing the code to
never block.

Never block
-----------

In essence, the Ert GUI code should never ever block. This is already
the case in most of the code-base. The exceptions are the aforementioned
QDialogs and QMessageBoxes. The other is whenever the GUI needs to
interact with some more Pythonic parts of Ert. Over time we wish for the
GUI to interact with Ert over some other channel, rather than calling
functions directly.

Even if we get to a point where Ert GUI and Ert proper interact over the
network, we will still need to be smart about Qt. It may be tempting to
implement starting processes and interacting with network sockets using
the Pythonic ``subprocess`` and ``socket``/``websockets`` modules. Doing
so would require the GUI to have two event handlers: One owned by Qt to
handle the GUI, and another one to handle the ``subprocess`` and/or
``socket``/``websockets`` using eg. ``asyncio``. This would be an error,
as the complexity of dealing with two systems that both think they have
the sole ownership of the process would be great and cause a lot of
headache when debugging.

Instead, we should take more advantage of Qt. Qt is not just a GUI
framework, but a whole application framework. It already contains
modules for dealing with subprocesses using ``QtCore.QProcess`` ,
sockets using ``QtNetwork.QTcpSocket`` and/or websockets using
``QtWebSockets.QWebSocket``. These components interact with the Qt event
system and can be used to implement what is required for dealing with
Ert. Components that do not have a Qt equivalent, like eg. ZeroMQ,
should be wrapped in a class that inherits from ``QObject`` and ran in a
separate Qt thread. This object should never block and instead poll.
This way, Qt remains in control, with the Python-based component
correctly being a second-class citizen in the system.

In essence, the GUI code must never ever block.
