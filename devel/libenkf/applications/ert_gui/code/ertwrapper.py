
from ctypes import *
import ctypes.util
import atexit
import re
import sys

class ErtWrapper:
    """Wraps the functionality of ERT using ctypes"""

    def __init__(self, site_config="/project/res/etc/ERT/Config/site-config", enkf_config="/private/jpb/EnKF/Testcases/SimpleEnKF/enkf_config", enkf_so="/private/jpb/EnKF/"):
        self.__loadLibraries__(enkf_so)

        self.pattern = re.compile("(?P<return>\w+) +(?P<function>[a-zA-Z]\w*) *[(](?P<arguments>[a-zA-Z0-9_, ]*)[)]")

        #bootstrap
        self.main = self.enkf.enkf_main_bootstrap(site_config, enkf_config)
        print "\nBootstrap complete!"
        
        self.plot_config = self.getErtPointer("enkf_main_get_plot_config")
        self.analysis_config = self.getErtPointer("enkf_main_get_analysis_config")
        self.ecl_config = self.getErtPointer("enkf_main_get_ecl_config")
        self.site_config = self.getErtPointer("enkf_main_get_site_config")
        self.ensemble_config = self.getErtPointer("enkf_main_get_ensemble_config")
        self.model_config = self.getErtPointer("enkf_main_get_model_config")
        self.logh = self.getErtPointer("enkf_main_get_logh")

        self.initializeTypes()
        
        atexit.register(self.cleanup)
        

        self.add_fixed_length_schedule_kw = ["item1", "item2"]
        self.schedule_prediction_file = "Missing???"

        self.dbase_type = "BLOCK_FS"
        self.enspath = "storage"
        self.select_case = "some_case"
        self.update_log_path = "one path"

        self.history_source = "REFCASE_HISTORY"
        self.obs_config = "..."

        self.pre_clear_runpath = True
        self.delete_runpath = "0 - 10, 12, 15, 20"
        self.keep_runpath = "0-15, 18, 20"

        self.license_path = "/usr"
        self.case_table = "..."

        self.run_template = [["...", ".....", "asdf:asdf asdfasdf:asdfasdf"], ["other", "sdtsdf", ".as.asdfsdf"]]
        self.forward_model = [["MY_RELPERM_SCRIPT", "Arg1<some> COPY(asdfdf)"]]

        self.registered_types = {}
        self.registerType("void", None)
        self.registerType("int", ctypes.c_int)
        self.registerType("bool", ctypes.c_int)
        self.registerType("long", ctypes.c_long)
        self.registerType("char", ctypes.c_char_p)
        self.registerType("float", ctypes.c_float)
        self.registerType("double", ctypes.c_double)
        self.registerType("ref", ctypes.c_void_p)


    def __loadLibraries__(self, prefix):
        """Load libraries that are required by ERT and ERT itself"""
        CDLL("libblas.so", RTLD_GLOBAL)
        CDLL("liblapack.so", RTLD_GLOBAL)
        CDLL("libz.so", RTLD_GLOBAL)

        self.util = CDLL(prefix + "libutil/slib/libutil.so", RTLD_GLOBAL)
        CDLL(prefix + "libecl/slib/libecl.so", RTLD_GLOBAL)
        CDLL(prefix + "libsched/slib/libsched.so", RTLD_GLOBAL)
        CDLL(prefix + "librms/slib/librms.so", RTLD_GLOBAL)
        CDLL(prefix + "libconfig/slib/libconfig.so", RTLD_GLOBAL)
        self.job_queue = CDLL(prefix + "libjob_queue/slib/libjob_queue.so", RTLD_GLOBAL)

        self.enkf = CDLL(prefix + "libenkf/slib/libenkf.so", RTLD_GLOBAL)

        self.enkf.enkf_main_install_SIGNALS()
        self.enkf.enkf_main_init_debug( "/usr/bin/python" )

    def registerType(self, type, value):
        """Register a type against a legal ctypes type"""
        self.registered_types[type] = value

    def _parseType(self, type):
        """Convert a prototype definition type from string to a ctypes legal type."""
        type = type.strip()

        if self.registered_types.has_key(type):
            return self.registered_types[type]
        else:
            return getattr(ctypes, type)

    def prototype(self, prototype, lib=None):
        """
        Provides the same functionality as setTypes but in a different way.
        prototype is a string formatted like this:

            "type functionName(type,type,type)"

        where type is a type available to ctypes
        Some type are automatically converted:
            int -> c_int
            long -> c_long
            char -> c_char_p
            bool -> c_int
            void -> None
            double -> c_double
            float -> c_float
            ref -> c_void_p (typically used for '&variable_name' situations (by reference))

        if lib is None lib defaults to the enkf library
        """
        if lib is None:
            lib = self.enkf

        match = re.match(self.pattern, prototype)
        if not match:
            sys.stderr.write("Illegal prototype definition: %s\n" % (prototype))
        else:
            restype = match.groupdict()["return"]
            functioname = match.groupdict()["function"]
            arguments = match.groupdict()["arguments"].split(",")
            #print restype, functioname, arguments

            func = getattr(lib, functioname)
            func.restype = self._parseType(restype)
            func.argtypes = [self._parseType(arg) for arg in arguments]
            #print func, func.restype, func.argtypes


    def setTypes(self, function, restype = c_long, argtypes = None, library = None, selfpointer = True):
        """
        Set the return and argument types of a ERT function.
        Since all methods need a pointer, this is already defined as c_long.
        library defaults to the enkf library
        """
        if library is None:
            library = self.enkf

        if argtypes is None:
            argtypes = []

        func = getattr(library, function)
        func.restype = restype

        if isinstance(argtypes, list):
            if selfpointer:
                args = [c_long]
                args.extend(argtypes)
            else:
                args = argtypes
            func.argtypes = args
        else:
            if selfpointer:
                func.argtypes = [c_long, argtypes]
            else:
                func.argtypes = [argtypes]


        #print "Setting: " + str(func.restype) + " " + function + "( " + str(func.argtypes) + " ) "
        return func


    def setAttribute(self, attribute, value):
        #print "set " + attribute + ": " + str(getattr(self, attribute)) + " -> " + str(value)
        setattr(self, attribute, value)

    def getAttribute(self, attribute):
        #print "get " + attribute + ": " + str(getattr(self, attribute))
        return getattr(self, attribute)

    def initializeTypes(self):
        self.setTypes("stringlist_iget", c_char_p, c_int, library = self.util)
        self.setTypes("stringlist_alloc_new", library = self.util, selfpointer=False)
        self.setTypes("stringlist_append_copy", None, c_char_p, library = self.util)
        self.setTypes("stringlist_get_size", c_int, library = self.util)
        self.setTypes("stringlist_free", None, library = self.util)

        self.setTypes("hash_iter_alloc", library = self.util)
        self.setTypes("hash_iter_get_next_key", c_char_p, library = self.util)
        self.setTypes("hash_get", c_char_p, library = self.util)
        self.setTypes("hash_get_int", c_int, library = self.util)
        self.setTypes("hash_iter_free", None, library = self.util)
        self.setTypes("hash_iter_is_complete", c_int, library = self.util)

        self.setTypes("subst_list_get_size", c_int, library = self.util)
        self.setTypes("subst_list_iget_key", c_char_p, c_int, library = self.util)
        self.setTypes("subst_list_iget_value", c_char_p, c_int, library = self.util)

        self.setTypes("bool_vector_alloc", c_long, [c_int, c_int], library = self.util, selfpointer=False)
        self.setTypes("bool_vector_iset", c_long, [c_int, c_int], library = self.util)
        self.setTypes("bool_vector_get_ptr", library = self.util)
        self.setTypes("bool_vector_free", None, library = self.util)

        self.setTypes("enkf_main_free", None)

        
    def getStringList(self, stringlistpointer, free_after_use=False):
        """Retrieve a list of strings"""
        if stringlistpointer == 0:
            return []

        result = []

        numberOfStrings = self.util.stringlist_get_size(stringlistpointer)

        for index in range(numberOfStrings):
            result.append(self.util.stringlist_iget(stringlistpointer, index))

        if free_after_use:
            self.freeStringList(stringlistpointer)

        return result

    def createStringList(self, list):
        """Creates a new string list from the specified list. Remember to free the list after use."""
        sl = self.util.stringlist_alloc_new()

        for item in list:
            self.util.stringlist_append_copy(sl , item)

        return sl


    def freeStringList(self, stringlistpointer):
        """Must be used if the stringlist was allocated on the python side"""
        self.util.stringlist_free(stringlistpointer)


    def getHash(self, hashpointer, intValue = False, return_type=c_char_p):
        """Retrieves a hash as a list of 2 element lists"""
        if hashpointer == 0:
            return []

        hash_iterator = self.util.hash_iter_alloc(hashpointer)
        self.setTypes("hash_get", return_type, library = self.util)

        result = []
        while not self.util.hash_iter_is_complete(hash_iterator):
            key   = self.util.hash_iter_get_next_key(hash_iterator)

            if not intValue:
                value = self.util.hash_get(hashpointer, key)
            else:
                value = self.util.hash_get_int(hashpointer, key)
                #print "%s -> %d" % (key , value)

            result.append([key, str(value)])

        self.util.hash_iter_free(hash_iterator)
        #print result
        return result

    def getSubstitutionList(self, substlistpointer):
        """Retrieves a substitution list as a list of 2 element lists"""
        size = self.util.subst_list_get_size(substlistpointer)

        result = []
        for index in range(size):
            key = self.util.subst_list_iget_key(substlistpointer, index)
            value = self.util.subst_list_iget_value(substlistpointer, index)
            result.append([key, value])

        return result

    def getErtPointer(self, function):
        """Returns a pointer from ERT as a c_long (64-bit support)"""
        func = getattr(self.enkf, function)
        func.restype = c_long
        return func(self.main)

    def createBoolVector(self, size, list):
        """Allocates a bool vector"""
        mask = self.util.bool_vector_alloc(size , False)

        for index in list:
            self.util.bool_vector_iset(mask, index, True)

        return mask

    def getBoolVectorPtr(self, mask):
        """Returns the pointer to a bool vector"""
        return self.util.bool_vector_get_ptr(mask)

    def freeBoolVector(self, mask):
        """Frees an allocated bool vector"""
        self.util.bool_vector_free(mask)

    def cleanup(self):
        """Called at atexit to clean up before shutdown"""
        print "Calling enkf_main_free()"
        self.enkf.enkf_main_free(self.main)

def testPrototypes():
    import ertwrapper
    import local
    site_config = "/project/res/etc/ERT/Config/site-config"
    enkf_config = local.enkf_config
    enkf_so     = local.enkf_so
    ert = ertwrapper.ErtWrapper(site_config = site_config, enkf_config = enkf_config, enkf_so = enkf_so)

    ert.prototype("c_char_p subst_list_iget_value(c_long, c_int)", ert.util)
    ert.prototype("void bool_vector_free (long)", ert.util)
    ert.prototype("void bool_vector_free (c_long)", ert.util)
    ert.prototype("char bool_vector_free (bool)", ert.util)
    #ert.prototype("int fungus()")
    #ert.prototype("int fungus(int, char_p, boo)")
    #ert.prototype("int fungus(int, char_p, boo")

if __name__ == "__main__":
    testPrototypes()

