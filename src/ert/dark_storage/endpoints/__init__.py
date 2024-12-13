from fastapi import APIRouter

from .compute.misfits import router as misfits_router
from .ensembles import router as ensembles_router
from .experiments import router as experiments_router
from .records import router as records_router
from .updates import router as updates_router

router = APIRouter()
router.include_router(experiments_router)
router.include_router(ensembles_router)
router.include_router(records_router)
router.include_router(updates_router)
router.include_router(misfits_router)
