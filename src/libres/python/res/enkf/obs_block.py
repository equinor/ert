from cwrap import BaseCClass
from res import ResPrototype
from res.util import Matrix


class ObsBlock(BaseCClass):
    TYPE_NAME = "obs_block"

    _alloc = ResPrototype(
        "void*  obs_block_alloc(char*, int, matrix, bool, double)", bind=False
    )
    _free = ResPrototype("void   obs_block_free(obs_block)")
    _total_size = ResPrototype("int    obs_block_get_size( obs_block )")
    _active_size = ResPrototype("int    obs_block_get_active_size( obs_block )")
    _iset = ResPrototype("void   obs_block_iset( obs_block , int , double , double)")
    _iget_value = ResPrototype("double obs_block_iget_value( obs_block , int)")
    _iget_std = ResPrototype("double obs_block_iget_std( obs_block , int)")
    _get_obs_key = ResPrototype("char* obs_block_get_key( obs_block )")
    _iget_is_active = ResPrototype("bool obs_block_iget_is_active( obs_block , int)")

    def __init__(self, obs_key, obs_size, global_std_scaling=1.0):
        error_covar = None
        error_covar_owner = False
        c_pointer = self._alloc(
            obs_key, obs_size, error_covar, error_covar_owner, global_std_scaling
        )
        super(ObsBlock, self).__init__(c_pointer)

    def totalSize(self):
        return self._total_size()

    def activeSize(self):
        return self.active()

    def active(self):
        return self._active_size()

    def __len__(self):
        """Returns the total size"""
        return self.totalSize()

    def is_active(self, index):
        return self._iget_is_active(index)

    def get_obs_key(self):
        return self._get_obs_key()

    def __setitem__(self, index, value):
        if len(value) != 2:
            raise TypeError(
                "The value argument must be a two element tuple: (value , std)"
            )
        d, std = value

        if isinstance(index, int):
            if index < 0:
                index += len(self)
            if 0 <= index < len(self):
                self._iset(index, d, std)
            else:
                raise IndexError(
                    "Invalid index: %d. Valid range: [0,%d)" % (index, len(self))
                )
        else:
            raise TypeError(
                "The index item must be integer, not %s." % str(type(index))
            )

    def __getitem__(self, index):
        if isinstance(index, int):
            if index < 0:
                index += len(self)
            if 0 <= index < len(self):
                value = self._iget_value(index)
                std = self._iget_std(index)
                return (value, std)
            else:
                raise IndexError(
                    "Invalid index:%d - valid range: [0,%d)" % (index, len(self))
                )
        else:
            raise TypeError(
                "The index item must be integer, not %s." % str(type(index))
            )

    def free(self):
        self._free()
