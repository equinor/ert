from fastapi import APIRouter
from .ensembles import router as ensembles_router
from .records import router as records_router
from .experiments import router as experiments_router
from .observations import router as observations_router
from .updates import router as updates_router
from .compute.misfits import router as misfits_router
from .responses import router as response_router

router = APIRouter()
router.include_router(experiments_router)
router.include_router(ensembles_router)
router.include_router(records_router)
router.include_router(observations_router)
router.include_router(updates_router)
router.include_router(misfits_router)
router.include_router(response_router)
