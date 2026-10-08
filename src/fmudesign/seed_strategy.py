from enum import StrEnum


class SeedStrategy(StrEnum):
    """How random-number state is assigned for Monte Carlo sampling.

    Both strategies use Latin Hypercube Sampling and honor configured
    correlations.

    JOINT (default):
        All parameters within a sensitivity are drawn together in one sampling
        call. The same random-number generator is reused across sensitivities,
        so changing or reordering an earlier sensitivity may also change later
        sensitivities. Iterative correlation can have the same downstream
        effect.
    INDEPENDENT:
        Each uncorrelated parameter and each correlation group uses a separate
        random-number generator derived from the base seed, sensitivity name,
        and parameter or group name. With a fixed base seed and sample size,
        changing one uncorrelated parameter leaves the other parameters and
        groups unchanged. A correlation group changes as one unit.
    """

    JOINT = "joint"
    INDEPENDENT = "independent"
