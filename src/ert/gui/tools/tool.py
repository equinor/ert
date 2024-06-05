from abc import abstractmethod
from typing import Optional

from qtpy.QtCore import QObject
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QAction


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
        self.__parent: Optional[QObject] = None
        self.__enabled = enabled
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

    @abstractmethod
    def trigger(self) -> None:
        pass

    def setParent(self, a0: Optional[QObject]) -> None:
        self.__parent = a0
        self.__action.setParent(a0)

    def parent(self) -> Optional[QObject]:
        return self.__parent

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
