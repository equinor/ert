#  Copyright (C) 2014  Statoil ASA, Norway. 
#   
#  The file 'enkf_fs.py' is part of ERT - Ensemble based Reservoir Tool. 
#   
#  ERT is free software: you can redistribute it and/or modify 
#  it under the terms of the GNU General Public License as published by 
#  the Free Software Foundation, either version 3 of the License, or 
#  (at your option) any later version. 
#   
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or 
#  FITNESS FOR A PARTICULAR PURPOSE.   
#   
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
#  for more details.
from cwrap import BaseCClass
from res.enkf import EnkfPrototype
from res.enkf import TimeMap, StateMap, RunArg
from res.enkf.enums import EnkfInitModeEnum
from ecl.util import PathFormat, StringList

class ErtRunContext(BaseCClass):
    TYPE_NAME = "ert_run_context"
    _alloc              = EnkfPrototype("void* ert_run_context_alloc( enkf_run_mode_enum , enkf_init_mode_enum, enkf_fs, enkf_fs , enkf_fs, bool_vector, path_fmt ,subst_list, int)", bind = False)
    _alloc_runpath_list = EnkfPrototype("stringlist_obj ert_run_context_alloc_runpath_list(bool_vector, path_fmt, subst_list, int)", bind = False)
    _alloc_runpath      = EnkfPrototype("char* ert_run_context_alloc_runpath(int, path_fmt, subst_list, int)", bind = False)
    _get_size           = EnkfPrototype("int ert_run_context_get_size( ert_run_context )")
    _free               = EnkfPrototype("void ert_run_context_free( ert_run_context )")
    _iget               = EnkfPrototype("run_arg_ref ert_run_context_iget_arg( ert_run_context , int)")
    _iens_get           = EnkfPrototype("run_arg_ref ert_run_context_iens_get_arg( ert_run_context , int)")
    _get_id             = EnkfPrototype("char* ert_run_context_get_id( ert_run_context )")
    _get_mask           = EnkfPrototype("bool_vector_ref ert_run_context_get_iactive( ert_run_context )")
    _get_iter           = EnkfPrototype("int ert_run_context_get_iter( ert_run_context )")
    _get_target_fs      = EnkfPrototype("enkf_fs_ref ert_run_context_get_update_target_fs( ert_run_context )")
    _get_result_fs      = EnkfPrototype("enkf_fs_ref ert_run_context_get_result_fs( ert_run_context )")
    
    def __init__(self , run_type , init_fs , result_fs, target_fs , mask , path_fmt , subst_list , itr, init_mode = EnkfInitModeEnum.INIT_CONDITIONAL):
        c_ptr = self._alloc( run_type, init_mode, init_fs , result_fs, target_fs, mask , path_fmt , subst_list, itr)
        super(ErtRunContext, self).__init__(c_ptr)

        # The C object ert_run_context uses a shared object for the
        # path_fmt, mask and subst_list objects. We therefor hold on
        # to a reference here - to inhibt Python GC of these objects.
        self._mask = mask
        self._path_fmt = path_fmt
        self._subst_list = subst_list

        

    def __len__(self):
        return self._get_size()

    def __getitem__(self , index):
        if isinstance(index, int):
            if 0 <= index < len(self):
                run_arg = self._iget(index)
                run_arg.setParent( self )
                return run_arg
            else:
                raise IndexError("Index:%d invalid. Legal range: [0,%d)" % (index , len(self)))
        else:
            raise TypeError("Invalid type - expetected integer")

    def iensGet(self , iens):
        run_arg = self._iens_get(iens)
        if run_arg is not None:
            run_arg.setParent(self)
            return run_arg
        else:
            raise ValueError("Run context does not have run argument for iens:%d" % iens)

    def free(self):
        self._free()

    def __repr__(self):
        return 'ErtRunContext(size = %d) %s' % (len(self), self._ad_str())

    @classmethod
    def createRunpathList(cls, mask, runpath_fmt, subst_list, iter=0):
        """ @rtype: ecl.util.stringlist.StringList """
        return cls._alloc_runpath_list(mask, runpath_fmt, subst_list, iter)


    @classmethod
    def createRunpath(cls, iens , runpath_fmt, subst_list, iter=0):
        """ @rtype: str """
        return cls._alloc_runpath(iens, runpath_fmt, subst_list, iter)

    
    def get_id(self):
        return self._get_id( )


    def get_mask(self):
        return self._get_mask( )


    def get_iter(self):
        return self._get_iter( )


    def get_target_fs(self):
        return self._get_target_fs( )


    def get_result_fs(self):
        return self._get_result_fs( )
