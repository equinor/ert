import os
from ert.util import IntVector
from ert.enkf.enums import ErtImplType, EnkfStateType, EnkfFieldFileFormatEnum
from ert.enkf.data import EnkfNode
from ert_gui.shell import ShellFunction, assertConfigLoaded, extractFullArgument, getPossibleFilenameCompletions



class Export(ShellFunction):
    DEFAULT_EXPORT_PATH = "export/%s/%s_%%d"

    def __init__(self , shell_context):
        super(Export , self).__init__("export" , shell_context)
        default_path = Export.DEFAULT_EXPORT_PATH % ("{KEY}" , "{KEY}")
        self.addHelpFunction("FIELD" , "<keyword>   [%s]  [1,4,7-10]" % default_path , "Export parameters; path and realisations in [...] are optional.")
        

        
    @assertConfigLoaded
    def do_FIELD(self , line):
        arguments = self.splitArguments(line)
        ens_config = self.ert().ensembleConfig()
        key = arguments[0]
        if key in ens_config:
            config_node = ens_config[key]
            if config_node.getImplementationType() == ErtImplType.FIELD:
                if len(arguments) >= 2:
                    path_fmt = arguments[1]
                else:
                    path_fmt = Export.DEFAULT_EXPORT_PATH % (key , key) + ".grdecl"
                    
                if len(arguments) >= 3:
                    range_string = "".join( arguments[2:] )
                    iens_list = IntVector.active_list( range_string )
                else:
                    ens_size = self.ert().getEnsembleSize()
                    iens_list = IntVector.createRange(0 , ens_size , 1 )

                fs_manager = self.ert().getEnkfFsManager( )
                fs = fs_manager.getCurrentFileSystem()
                init_file = self.ert().fieldInitFile( config_node )
                if init_file:
                    print "Using init file:%s" % init_file
                EnkfNode.exportMany( config_node , path_fmt , fs , iens_list , arg = init_file)
            else:
                self.lastCommandFailed("Node:%s is not of type FIELD" % key)
        else:
            self.lastCommandFailed("No such FIELD node: %s" % key)
