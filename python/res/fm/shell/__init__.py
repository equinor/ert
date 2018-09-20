from .shell import Shell

def prefix_path(input_path):
  Shell.prefix_path(input_path)

def symlink(target , link_name):
  Shell.symlink(target , link_name)

def mkdir(path):
  Shell.mkdir(path)

def moveFile(src_file , target):
  Shell.moveFile(src_file , target)

def deleteFile(filename):
  Shell.deleteFile(filename)

def deleteDirectory(path):
  Shell.deleteDirectory(path)

def copyDirectory(src_path , target_path ):
  Shell.copyDirectory(src_path , target_path )

def copyFile(src , target ):
  Shell.copyFile(src , target )


