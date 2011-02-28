#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'ecl.py' is part of ERT - Ensemble based Reservoir Tool. 
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

import pwd
import grp
import os
import shutil
import os.path
import re
import stat

DIR_MODE = 0755
EXE_MODE = 0755
REG_MODE = 0644

def msg( verbose , text , arg):
    text_width = 20
    if verbose:
        pad_text = text + (text_width - len( text)) * "." + ": "
        print pad_text + arg
        

def update_mode( path , mode , user , group):
    os.chmod( path , mode )
    
    if user:   # Only applicable for root:
        user_info = pwd.getpwnam( user )
        uid = user_info[ 2 ]
        os.chown( path , uid , -1 )
        
    if group:
        group_info = grp.getgrnam( group )
        gid = group_info[ 2 ]
        os.chown( path , -1 , gid )
    



class File:
    
    def __init__( self , src , create_path = None , target_name = None ):
        self.src = src
        self.create_path = create_path
        self.name = os.path.basename( self.src )
        if target_name:
            self.target_name = target_name
        else:
            self.target_name = self.name


    def install( self , src_root , target_root , verbose , user , group):
        if self.create_path:
            target_path = target_root
            for path in self.create_path.split("/"):
                target_path += "/%s" % path
                if not os.path.exists( target_path ):
                    os.makedirs( target_path , DIR_MODE )
                    msg(verbose , "Creating directory" , target_path)
                update_mode( target_path , DIR_MODE , user, group)
            target_file = "%s/%s/%s" % (target_root , self.create_path , self.target_name)
        else:
            target_file = "%s/%s" % (target_root ,  self.target_name)
        src_file = "%s/%s" % (src_root , self.src)
        msg( verbose , "Copying file" , "%s -> %s" % (src_file , target_file))
        shutil.copyfile( src_file , target_file )
        if os.access( src_file , os.X_OK):
            update_mode( target_file , EXE_MODE , user , group)
        else:
            update_mode( target_file , REG_MODE , user , group )
        (target_base , ext) = os.path.splitext( target_file )
        if ext == ".py":
            msg( verbose , "Byte compiling" , target_file)
            py_compile.compile( target_file )
            pyc_file = target_base + ".pyc"
            chgrp( pyc_file , res_guid )
            chmod( pyc_file , data_mode )
            
                


class Install:
    def __init__( self , src_root ):
        self.src_root = src_root
        self.file_list = []


    def install(self, target_root , user = None , group = None , verbose = False):
        for file in self.file_list:
            file.install( self.src_root , target_root , verbose , user , group)


    def add_path( self , path , create_path = None , recursive = False):
        for entry in os.listdir( "%s/%s" % (self.src_root , path )):
            (base , ext) = os.path.splitext( entry )

            if ext == ".pyc":
                continue
            if ext == ".svn":
                continue
            

            full_path = "%s/%s/%s" % ( self.src_root , path , entry)
            mode = os.stat( full_path )[stat.ST_MODE]
            if stat.S_ISREG( mode ):
                file = File("%s/%s" % (path , entry), create_path = create_path)
                self.add_file( file )
            elif stat.S_ISDIR( mode ):
                if recursive:
                    self.add_path( "%s/%s" % (path , entry) , create_path = create_path , verbose = True )
                    

    def add_file( self , file ):
        self.file_list.append( file )
    


            
        
