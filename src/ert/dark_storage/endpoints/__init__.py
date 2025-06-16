from fastapi import APIRouter

from .compute.misfits import router as misfits_router
from .ensembles import router as ensembles_router
from .experiment_server import router as experiment_server_router
from .experiments import router as experiments_router
from .observations import router as observations_router
from .parameters import router as parameters_router
from .responses import router as responses_router
from .updates import router as updates_router

router = APIRouter()
router.include_router(experiments_router)
router.include_router(ensembles_router)
router.include_router(observations_router)
router.include_router(updates_router)
router.include_router(misfits_router)
router.include_router(parameters_router)
router.include_router(responses_router)
router.include_router(experiment_server_router)
