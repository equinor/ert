from PyQt6.QtWidgets import QFrame, QLabel, QSizePolicy

# Spacing between cards. The experiment settings and the summary below them
# both use three equally wide columns with this spacing so the cards line up.
CARD_SPACING = 12


def create_card() -> QFrame:
    card = QFrame()
    card.setObjectName("card")
    # Ignore the content width so that all columns get the same width.
    card.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    return card


def create_card_title(title: str) -> QLabel:
    label = QLabel(title)
    label.setStyleSheet("font-size: 16px; font-weight: bold;")
    return label
