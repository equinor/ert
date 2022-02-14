from cwrap import BaseCClass

from res import ResPrototype


class LocalObsdataNode(BaseCClass):
    TYPE_NAME = "local_obsdata_node"

    _alloc = ResPrototype("void* local_obsdata_node_alloc(char*)", bind=False)
    _free = ResPrototype("void  local_obsdata_node_free(local_obsdata_node)")
    _get_key = ResPrototype("char* local_obsdata_node_get_key(local_obsdata_node)")
    _get_active_list = ResPrototype(
        "active_list_ref local_obsdata_node_get_active_list(local_obsdata_node)"
    )

    def __init__(self, obs_key):
        if isinstance(obs_key, str):
            c_ptr = self._alloc(obs_key)
            if c_ptr:
                super().__init__(c_ptr)
            else:
                raise ArgumentError(
                    'Unable to construct LocalObsdataNode with key = "%s".' % obs_key
                )
        else:
            raise TypeError(
                "LocalObsdataNode needs string, not %s." % str(type(obs_key))
            )

    def key(self):
        return self._get_key()

    def getKey(self):
        return self.key()

    def free(self):
        self._free()

    def __repr__(self):
        return "LocalObsdataNode(key = %s) %s" % (self.key(), self._ad_str())

    def getActiveList(self):
        return self._get_active_list()
