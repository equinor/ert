from PyQt6.QtWidgets import QApplication

IS_EVEREST_APPLICATION = QApplication.applicationName().lower() == "everest"
