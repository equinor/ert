from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import create_engine
from .ert_adapter import ERT
from ert_shared.storage.model import Entities, Blobs
from contextlib import contextmanager

try:
    from ert_shared.version import version as __version__
except ImportError:
    __version__ = '0.0.0'

def clear_global_state():
    """Deletes the global ERT instance shared as a global variable in the
    ert_shared module. This is due to an exception that arrises when closing
    the ERT application when modules, Python objects and C-objects are removed.
    Over time the singleton instance of ERT should disappear and this function
    should be removed.
    """
    global ERT
    if ERT is None:
        return

    ERT._implementation = None
    ERT._enkf_facade = None
    ERT = None

class ErtStorage():
    def __init__(self):
        self.rdb_url = None
        self.blob_url = None
    
    def initialize(self, rdb_url="sqlite:///entities.db", blob_url="sqlite:///blobs.db"):
        self.rdb_url = rdb_url
        self.blob_url = blob_url
        rdb_engine = create_engine(rdb_url)
        blob_engine = create_engine(blob_url)    
        self._upgrade(rdb_engine=rdb_engine, blob_engine=blob_engine)
            
        self.RdbSession = sessionmaker(bind=rdb_engine)
        self.BlobSession = sessionmaker(bind=blob_engine)   
            
    def _upgrade(self, rdb_engine, blob_engine, pragma_foreign_keys=True):               
        if pragma_foreign_keys:
            rdb_engine.execute("pragma foreign_keys=on")
            blob_engine.execute("pragma foreign_keys=on")
        Entities.metadata.create_all(rdb_engine)        
        Blobs.metadata.create_all(blob_engine)        

ERT_STORAGE = ErtStorage()

