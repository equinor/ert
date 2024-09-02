from typing import Any, List, Tuple


class HasErtQueueOptions:
    def extract_ert_queue_options(
        self, queue_system: str, everest_to_ert_key_tuples: List[Tuple[str, str]]
    ) -> List[Tuple[str, str, Any]]:
        result = []
        for ever_key, ert_key in everest_to_ert_key_tuples:
            attribute = getattr(self, ever_key)
            if attribute is not None:
                result.append((queue_system, ert_key, attribute))
        return result
