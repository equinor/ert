from PyQt6.QtWidgets import QFrame, QSizePolicy


class Card(QFrame):
    # The experiment settings and the summary below them both use three
    # equally wide columns with this spacing so the cards line up.
    SPACING = 12

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("card")
        # Ignore the content width so that all columns get the same width.
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
