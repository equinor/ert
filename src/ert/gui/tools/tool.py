from PyQt6.QtCore import QObject
from PyQt6.QtGui import QAction, QIcon


class Tool:
    def __init__(
        self,
        name: str,
        icon: QIcon,
        enabled: bool = True,
        checkable: bool = False,
        popup_menu: bool = False,
    ) -> None:
        super().__init__()
        self.__icon = icon
        self.__name = name
        self.__parent: QObject | None = None
        self.__enabled = enabled
        self.__checkable = checkable
        self.__is_popup_menu = popup_menu

        self.__action = QAction(self.getIcon(), self.getName(), None)
        self.__action.setIconText(self.getName())
        self.__action.setEnabled(self.isEnabled())
        self.__action.setCheckable(checkable)
        self.__action.triggered.connect(self.trigger)

    def getIcon(self) -> QIcon:
        return self.__icon

    def getName(self) -> str:
        return self.__name

    def trigger(self) -> None:
        raise NotImplementedError

    def setParent(self, parent: QObject | None) -> None:
        self.__parent = parent
        self.__action.setParent(parent)

    def parent(self) -> QObject:
        return self.__parent or QObject()

    def isEnabled(self) -> bool:
        return self.__enabled

    def getAction(self) -> QAction:
        return self.__action

    def setVisible(self, visible: bool) -> None:
        self.__action.setVisible(visible)

    def setEnabled(self, enabled: bool) -> None:
        self.__action.setEnabled(enabled)

    def isPopupMenu(self) -> bool:
        return self.__is_popup_menu
