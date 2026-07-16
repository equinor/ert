from matplotlib import colormaps
from matplotlib.colors import TABLEAU_COLORS, to_hex

ALPHA_VALUE = 1.0

TABLEAU_10_COLOR_CYCLE = [
    (to_hex(color), ALPHA_VALUE) for color in TABLEAU_COLORS.values()
]

_okabe_ito = colormaps.get("okabe_ito")
# Fallback in case matplotlib version < 3.11
if _okabe_ito is None:
    OKABE_ITO_COLOR_CYCLE = [
        ("#E69F00", ALPHA_VALUE),  # Orange
        ("#56B4E9", ALPHA_VALUE),  # Sky Blue
        ("#009E73", ALPHA_VALUE),  # Bluish Green
        ("#A4A49B", ALPHA_VALUE),  # Yellow
        ("#0072B2", ALPHA_VALUE),  # Blue
        ("#D55E00", ALPHA_VALUE),  # Vermillion
        ("#CC79A7", ALPHA_VALUE),  # Reddish Purple
    ]
else:
    # Skipping black color at index 0
    # as this is only for specific lines
    # (e.g history, observations, mean etc)
    OKABE_ITO_COLOR_CYCLE = [
        (to_hex(_okabe_ito(index)), ALPHA_VALUE) for index in range(1, _okabe_ito.N)
    ]
COLOR_BREWER_COLOR_CYCLE = [
    ("#a6cee3", ALPHA_VALUE),
    ("#1f78b4", ALPHA_VALUE),
    ("#b2df8a", ALPHA_VALUE),
    ("#33a02c", ALPHA_VALUE),
    ("#fb9a99", ALPHA_VALUE),
    ("#e31a1c", ALPHA_VALUE),
    ("#fdbf6f", ALPHA_VALUE),
    ("#ff7f00", ALPHA_VALUE),
    ("#cab2d6", ALPHA_VALUE),
    ("#6a3d9a", ALPHA_VALUE),
    ("#ffff99", ALPHA_VALUE),
    ("#b15928", ALPHA_VALUE),
]
PALETTES_WITH_DESCRIPTIONS: dict[str, tuple[list[tuple[str, float]], str]] = {
    "Tableau 10 (default)": (
        TABLEAU_10_COLOR_CYCLE,
        (
            "The Matplotlib default palette."
            f"\nColorblind-safe palette ({len(TABLEAU_10_COLOR_CYCLE)} colors)"
        ),
    ),
    "Okabe Ito": (
        OKABE_ITO_COLOR_CYCLE,
        f"Colorblind-safe palette ({len(OKABE_ITO_COLOR_CYCLE)} colors)",
    ),
    "Color Brewer": (
        COLOR_BREWER_COLOR_CYCLE,
        (
            "Color palette from ColorBrewer "
            f"({len(COLOR_BREWER_COLOR_CYCLE)} colors). "
            "Good for qualitative data, but not colorblind-safe."
        ),
    ),
}
