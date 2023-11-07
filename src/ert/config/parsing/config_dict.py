from typing import Dict

from .context_values import ContextString
from .types import MaybeWithContext

ConfigDict = Dict[ContextString, MaybeWithContext]
