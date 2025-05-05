from .context_values import ContextString
from .types import MaybeWithContext

ConfigDict = dict[ContextString | str, MaybeWithContext]
