from collections import defaultdict
from itertools import count
from typing import DefaultDict, Dict, List, Optional, Tuple

import numpy as np
from numpy._typing import NDArray


# This cache can be used to prevent re-evaluation of forward models. Due to its
# simplicity it has some limitations:
#   - There is no limit on the number of cached entries.
#   - Searching in the cache is by brute-force, iterating over the entries.
# Both of these should not be an issue for the intended use with cases where the
# forward models are very expensive to compute: The number of cached entries is
# not expected to become prohibitively large.
class SimulatorCache:
    def __init__(self) -> None:
        # Stores the realization/controls key, together with an ID.
        self._keys: DefaultDict[int, List[Tuple[NDArray[np.float64], int]]] = (
            defaultdict(list)
        )
        # Store objectives and constraints by ID:
        self._objectives: Dict[int, NDArray[np.float64]] = {}
        self._constraints: Dict[int, NDArray[np.float64]] = {}

        # Generate unique ID's:
        self._counter = count()

    def add_simulation_results(
        self,
        sim_idx: int,
        real_id: int,
        control_values: NDArray[np.float64],
        objectives: NDArray[np.float64],
        constraints: Optional[NDArray[np.float64]],
    ):
        cache_id = next(self._counter)
        self._keys[real_id].append((control_values[sim_idx, :].copy(), cache_id))
        self._objectives[cache_id] = objectives[sim_idx, ...].copy()
        if constraints is not None:
            self._constraints[cache_id] = constraints[sim_idx, ...].copy()

    def find_key(
        self, real_id: int, control_vector: NDArray[np.float64]
    ) -> Optional[int]:
        # Brute-force search, premature optimization is the root of all evil:
        for cached_vector, cache_id in self._keys.get(real_id, []):
            if np.allclose(
                control_vector,
                cached_vector,
                rtol=0.0,
                atol=float(np.finfo(np.float32).eps),
            ):
                return cache_id
        return None

    def get_objectives(self, cache_id: int) -> NDArray[np.float64]:
        return self._objectives[cache_id]

    def get_constraints(self, cache_id: int) -> NDArray[np.float64]:
        return self._constraints[cache_id]
