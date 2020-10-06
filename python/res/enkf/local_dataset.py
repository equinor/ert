from cwrap import BaseCClass
from res import ResPrototype
from ecl.grid import EclRegion
from ecl.util.geometry import GeoRegion


class LocalDataset(BaseCClass):
    TYPE_NAME = "local_dataset"

    _alloc = ResPrototype("void* local_dataset_alloc(char*)", bind=False)
    _size = ResPrototype("int   local_dataset_get_size(local_dataset)")
    _has_key = ResPrototype("bool  local_dataset_has_key(local_dataset, char*)")
    _free = ResPrototype("void  local_dataset_free(local_dataset)")
    _name = ResPrototype("char* local_dataset_get_name(local_dataset)")
    _add_node = ResPrototype("void  local_dataset_add_node(local_dataset, char*)")
    _del_node = ResPrototype("void  local_dataset_del_node(local_dataset, char*)")
    _active_list = ResPrototype(
        "active_list_ref local_dataset_get_node_active_list(local_dataset, char*)"
    )

    def __init__(self, name):
        raise NotImplementedError("Class can not be instantiated directly!")

    def initEnsembleConfig(self, config):
        self.ensemble_config = config

    def __len__(self):
        """ @rtype: int """
        return self._size()

    def __contains__(self, key):
        """ @rtype: bool """
        return self._has_key(key)

    def __delitem__(self, key):
        assert isinstance(key, str)
        if key in self:
            self._del_node(key)
        else:
            raise KeyError('Unknown key "%s"' % key)

    def name(self):
        return self._name()

    def getName(self):
        """ @rtype: str """
        return self.name()

    def addNode(self, key):
        assert isinstance(key, str)
        if key in self.ensemble_config:
            if not self._has_key(key):
                self._add_node(key)
            else:
                raise KeyError('Tried to add existing data key "%s".' % key)
        else:
            raise KeyError('Tried to add data key "%s" not in ensemble.' % key)

    def addNodeWithIndex(self, key, index):
        assert isinstance(key, str)
        assert isinstance(index, int)

        self.addNode(key)
        active_list = self.getActiveList(key)
        active_list.addActiveIndex(index)

    def addRegion(self, key, region):
        assert isinstance(key, str)
        self.addNode(key)
        active_list = self.getActiveList(key)
        active_region = region.getActiveList()
        for i in active_region:
            active_list.addActiveIndex(i)

    def addField(self, key, ecl_region):
        assert isinstance(ecl_region, EclRegion)
        self.addRegion(str(key), ecl_region)

    def addSurface(self, key, geo_region):
        assert isinstance(geo_region, GeoRegion)
        self.addRegion(str(key), geo_region)

    def getActiveList(self, key):
        """ @rtype: ActiveList """
        if key in self:
            return self._active_list(key)
        else:
            raise KeyError('Local key "%s" not recognized.' % key)

    def free(self):
        self._free()

    def __repr__(self):
        return self._create_repr("name=%s, size=%d" % (self.name(), len(self)))
