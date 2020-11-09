from fastapi import APIRouter
from ert_shared.storage.endpoints import ensembles
from ert_shared.storage.endpoints import observations
from ert_shared.storage.endpoints import parameters
from ert_shared.storage.endpoints import realizations
from ert_shared.storage.endpoints import responses


router = APIRouter()
router.include_router(ensembles.router)
router.include_router(observations.router)
router.include_router(parameters.router)
router.include_router(realizations.router)
router.include_router(responses.router)
