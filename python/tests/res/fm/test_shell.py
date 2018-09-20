import unittest
import os
import os.path

from ecl.util.test import TestAreaContext
from tests import ResTest
from res.fm.shell import Shell



class ShellTest(ResTest):

    def setUp(self):
        if "DATA_ROOT" in os.environ:
            del os.environ["DATA_ROOT"]


    def test_symlink(self):
        with self.assertRaises(IOError):
            Shell.symlink( "target/does/not/exist" , "link")

        with TestAreaContext("symlink/test"):
            with open("target", "w") as fileH:
                fileH.write("target ...")

            Shell.symlink( "target" , "link")
            self.assertTrue( os.path.islink("link") )
            self.assertEqual( os.readlink("link") , "target")

            with open("target2", "w") as fileH:
                fileH.write("target ...")

            with self.assertRaises(OSError):
                Shell.symlink("target2" , "target")


            Shell.symlink("target2" , "link")
            self.assertTrue( os.path.islink("link") )
            self.assertEqual( os.readlink("link") , "target2")

            os.makedirs("root1/sub1/sub2")
            os.makedirs("root2/sub1/sub2")
            os.makedirs("run")

            Shell.symlink("../target" , "linkpath/link")
            self.assertTrue( os.path.isdir( "linkpath" ))
            self.assertTrue( os.path.islink( "linkpath/link"))

            Shell.symlink("../target" , "linkpath/link")
            self.assertTrue( os.path.isdir( "linkpath" ))
            self.assertTrue( os.path.islink( "linkpath/link"))


        with TestAreaContext("symlink/test2"):
            os.makedirs("path")
            with open("path/target", "w") as f:
                f.write("1234")

            Shell.symlink("path/target" , "link")
            self.assertTrue( os.path.islink( "link" ))
            self.assertTrue( os.path.isfile( "path/target" ))

            Shell.symlink("path/target" , "link")
            self.assertTrue( os.path.islink( "link" ))
            self.assertTrue( os.path.isfile( "path/target" ))
            with open("link") as f:
                s = f.read( )
                self.assertEqual( s, "1234" )




    def test_mkdir(self):
        with TestAreaContext("shell/mkdir"):
            with open("file", "w") as f:
                f.write("Hei")

            with self.assertRaises(OSError):
                Shell.mkdir( "file" )

            Shell.mkdir("path")
            self.assertTrue( os.path.isdir( "path"))
            Shell.mkdir("path")

            Shell.mkdir("path/subpath")
            self.assertTrue( os.path.isdir( "path/subpath"))


    def test_move_file(self):
        with TestAreaContext("shell/move_file"):
            with open("file", "w") as f:
                f.write("Hei")

            Shell.moveFile("file" , "file2")
            self.assertTrue( os.path.isfile("file2"))
            self.assertFalse( os.path.isfile("file"))

            with self.assertRaises(IOError):
                Shell.moveFile("file2" , "path/file2")

            Shell.mkdir("path")
            Shell.moveFile("file2" , "path/file2")
            self.assertTrue( os.path.isfile("path/file2"))
            self.assertFalse( os.path.isfile( "file2") )

            with self.assertRaises(IOError):
                Shell.moveFile("path" , "path2")

            with self.assertRaises(IOError):
                Shell.moveFile("not_existing" , "target")

            with open("file2","w") as f:
                f.write("123")

            Shell.moveFile("file2" , "path/file2")
            self.assertTrue( os.path.isfile("path/file2"))
            self.assertFalse( os.path.isfile( "file2") )

            Shell.mkdir("rms/ipl")
            with open("global_variables.ipl","w") as f:
                f.write("123")

            Shell.moveFile("global_variables.ipl" , "rms/ipl/global_variables.ipl")

    def test_move_file_into_folder_file_exists(self):
        with TestAreaContext("shell/test_move_file_into_folder_file_exists"):
            Shell.mkdir("dst_folder")
            with open("dst_folder/file", "w") as f:
                f.write("old")

            with open("file", "w") as f:
                f.write("new")

            with open("dst_folder/file", "r") as f:
                content = f.read()
                self.assertEqual(content, "old")

            Shell.moveFile("file", "dst_folder")
            with open("dst_folder/file", "r") as f:
                content = f.read()
                self.assertEqual(content, "new")

            self.assertFalse(os.path.exists("file"))

    def test_move_pathfile_into_folder(self):
        with TestAreaContext("shell/test_move_pathfile"):
            Shell.mkdir("dst_folder")
            Shell.mkdir("source1/source2/")
            with open("source1/source2/file", "w") as f:
                f.write("stuff")


            Shell.moveFile("source1/source2/file", "dst_folder")
            with open("dst_folder/file", "r") as f:
                content = f.read()
                self.assertEqual(content, "stuff")

            self.assertFalse(os.path.exists("source1/source2/file"))

    def test_move_pathfile_into_folder_file_exists(self):
        with TestAreaContext("shell/test_move_pathfile"):
            Shell.mkdir("dst_folder")
            Shell.mkdir("source1/source2/")
            with open("source1/source2/file", "w") as f:
                f.write("stuff")

            with open("dst_folder/file", "w") as f:
                f.write("garbage")

            Shell.moveFile("source1/source2/file", "dst_folder")
            with open("dst_folder/file", "r") as f:
                content = f.read()
                self.assertEqual(content, "stuff")

            self.assertFalse(os.path.exists("source1/source2/file"))


    def test_delete_file(self):
        with TestAreaContext("delete/file"):
            Shell.mkdir("pathx")
            with self.assertRaises(IOError):
                Shell.deleteFile( "pathx" )

            # deleteFile which does not exist - is silently ignored
            Shell.deleteFile("does/not/exist")


            with open("file" , "w") as f:
                f.write("hei")
            Shell.symlink("file", "link")
            self.assertTrue(os.path.islink("link"))

            Shell.deleteFile("file")
            self.assertFalse( os.path.isfile( "file" ))
            self.assertTrue( os.path.islink("link"))
            Shell.deleteFile("link")
            self.assertFalse( os.path.islink("link"))



    def test_delete_directory(self):
        with TestAreaContext("delete/directory"):
            # deleteDriecteory which does not exist - is silently ignored
            Shell.deleteDirectory("does/not/exist")

            with open("file" , "w") as f:
                f.write("hei")

            with self.assertRaises(IOError):
                Shell.deleteDirectory("file")


            Shell.mkdir("link_target/subpath")
            with open("link_target/link_file" , "w") as f:
                f.write("hei")

            Shell.mkdir("path/subpath")
            with open("path/file" , "w") as f:
                f.write("hei")

            with open("path/subpath/file" , "w") as f:
                f.write("hei")

            Shell.symlink( "../link_target" , "path/link")
            Shell.deleteDirectory("path")
            self.assertFalse( os.path.exists( "path" ))
            self.assertTrue( os.path.exists("link_target/link_file"))

    def test_copy_directory_error(self):
        with self.assertRaises(IOError):
            Shell.copyDirectory("does/not/exist" , "target")

        with TestAreaContext("copy/directory"):
            with open("file" , "w") as f:
                f.write("hei")

            with self.assertRaises(IOError):
                Shell.copyDirectory("hei" , "target")

    def test_DATA_ROOT(self):
        with TestAreaContext("copy/directory"):

            Shell.mkdir("path/subpath")
            with open("path/subpath/file" , "w") as f:
                f.write("1")

            os.environ["DATA_ROOT"] = "path"
            Shell.mkdir("target/sub")
            Shell.copyDirectory("subpath" , "target/sub")
            self.assertTrue( os.path.exists( "target/sub/subpath" ))
            self.assertTrue( os.path.exists( "target/sub/subpath/file" ))

            os.makedirs( "file_target")
            Shell.copyFile( "subpath/file" , "file_target")
            self.assertTrue( os.path.isfile( "file_target/file" ))

            Shell.copyFile( "subpath/file" , "subpath/file")
            with open("subpath/file") as f:
                v = int(f.read())
                self.assertEqual( v, 1 )

            with open("path/subpath/file" , "w") as f:
                f.write("2")
            Shell.copyFile( "subpath/file" , "subpath/file")
            with open("subpath/file") as f:
                v = int(f.read())
                self.assertEqual( v, 2 )

            Shell.symlink( "subpath/file" , "file_link")
            self.assertTrue( os.path.isfile( "file_link" ))
            self.assertTrue( os.path.islink( "file_link" ))
            self.assertEqual( os.readlink( "file_link" ) , "path/subpath/file")
            Shell.deleteDirectory( "subpath" )
            self.assertFalse( os.path.isdir( "path/subpath") )


    def test_copy_file(self):
        with TestAreaContext("copy/file"):
            with self.assertRaises(IOError):
                Shell.copyFile("does/not/exist" , "target")

            Shell.mkdir("path")
            with self.assertRaises(IOError):
                Shell.copyFile("path" , "target")


            with open("file1" , "w") as f:
                f.write("hei")

            Shell.copyFile("file1" , "file2")
            self.assertTrue( os.path.isfile("file2") )

            Shell.copyFile("file1" , "path")
            self.assertTrue( os.path.isfile("path/file1") )

            Shell.copyFile("file1" , "path2/file1")
            self.assertTrue( os.path.isfile("path2/file1") )


            with TestAreaContext("copy/file2"):
                Shell.mkdir("root/sub/path")

                with open("file" , "w") as f:
                    f.write("Hei ...")

                Shell.copyFile("file" , "root/sub/path/file")
                self.assertTrue( os.path.isfile( "root/sub/path/file"))


    def test_copy_file2(self):
        with TestAreaContext("copy/file2"):
            Shell.mkdir("rms/output")

            with open("file.txt" , "w") as f:
                f.write("Hei")

            Shell.copyFile("file.txt" , "rms/output/")
            self.assertTrue( os.path.isfile( "rms/output/file.txt" ))


if __name__ == "__main__":
    unittest.main()
