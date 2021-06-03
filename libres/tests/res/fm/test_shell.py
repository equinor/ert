import unittest
import os
import os.path
import contextlib

from ecl.util.test import TestAreaContext
from tests import ResTest
from res.fm.shell import *
from pytest import MonkeyPatch


@contextlib.contextmanager
def pushd(path):
    cwd0 = os.getcwd()
    os.chdir(path)

    yield

    os.chdir(cwd0)


class ShellTest(ResTest):
    def setUp(self):
        self.monkeypatch = MonkeyPatch()

    def tearDown(self):
        self.monkeypatch.undo()

    def test_symlink(self):
        with self.assertRaises(IOError):
            symlink("target/does/not/exist", "link")

        with TestAreaContext("symlink/test"):
            with open("target", "w") as fileH:
                fileH.write("target ...")

            symlink("target", "link")
            self.assertTrue(os.path.islink("link"))
            self.assertEqual(os.readlink("link"), "target")

            with open("target2", "w") as fileH:
                fileH.write("target ...")

            with self.assertRaises(OSError):
                symlink("target2", "target")

            symlink("target2", "link")
            self.assertTrue(os.path.islink("link"))
            self.assertEqual(os.readlink("link"), "target2")

            os.makedirs("root1/sub1/sub2")
            os.makedirs("root2/sub1/sub2")
            os.makedirs("run")

            symlink("../target", "linkpath/link")
            self.assertTrue(os.path.isdir("linkpath"))
            self.assertTrue(os.path.islink("linkpath/link"))

            symlink("../target", "linkpath/link")
            self.assertTrue(os.path.isdir("linkpath"))
            self.assertTrue(os.path.islink("linkpath/link"))

        with TestAreaContext("symlink/test2"):
            os.makedirs("path")
            with open("path/target", "w") as f:
                f.write("1234")

            symlink("path/target", "link")
            self.assertTrue(os.path.islink("link"))
            self.assertTrue(os.path.isfile("path/target"))

            symlink("path/target", "link")
            self.assertTrue(os.path.islink("link"))
            self.assertTrue(os.path.isfile("path/target"))
            with open("link") as f:
                s = f.read()
                self.assertEqual(s, "1234")

    def test_mkdir(self):
        with TestAreaContext("shell/mkdir"):
            with open("file", "w") as f:
                f.write("Hei")

            with self.assertRaises(OSError):
                mkdir("file")

            mkdir("path")
            self.assertTrue(os.path.isdir("path"))
            mkdir("path")

            mkdir("path/subpath")
            self.assertTrue(os.path.isdir("path/subpath"))

    def test_move_file(self):
        with TestAreaContext("shell/move_file"):
            with open("file", "w") as f:
                f.write("Hei")

            move_file("file", "file2")
            self.assertTrue(os.path.isfile("file2"))
            self.assertFalse(os.path.isfile("file"))

            with self.assertRaises(IOError):
                move_file("file2", "path/file2")

            mkdir("path")
            move_file("file2", "path/file2")
            self.assertTrue(os.path.isfile("path/file2"))
            self.assertFalse(os.path.isfile("file2"))

            with self.assertRaises(IOError):
                move_file("path", "path2")

            with self.assertRaises(IOError):
                move_file("not_existing", "target")

            with open("file2", "w") as f:
                f.write("123")

            move_file("file2", "path/file2")
            self.assertTrue(os.path.isfile("path/file2"))
            self.assertFalse(os.path.isfile("file2"))

            mkdir("rms/ipl")
            with open("global_variables.ipl", "w") as f:
                f.write("123")

            move_file("global_variables.ipl", "rms/ipl/global_variables.ipl")

    def test_move_file_into_folder_file_exists(self):
        with TestAreaContext("shell/test_move_file_into_folder_file_exists"):
            mkdir("dst_folder")
            with open("dst_folder/file", "w") as f:
                f.write("old")

            with open("file", "w") as f:
                f.write("new")

            with open("dst_folder/file", "r") as f:
                content = f.read()
                self.assertEqual(content, "old")

            move_file("file", "dst_folder")
            with open("dst_folder/file", "r") as f:
                content = f.read()
                self.assertEqual(content, "new")

            self.assertFalse(os.path.exists("file"))

    def test_move_pathfile_into_folder(self):
        with TestAreaContext("shell/test_move_pathfile"):
            mkdir("dst_folder")
            mkdir("source1/source2/")
            with open("source1/source2/file", "w") as f:
                f.write("stuff")

            move_file("source1/source2/file", "dst_folder")
            with open("dst_folder/file", "r") as f:
                content = f.read()
                self.assertEqual(content, "stuff")

            self.assertFalse(os.path.exists("source1/source2/file"))

    def test_move_pathfile_into_folder_file_exists(self):
        with TestAreaContext("shell/test_move_pathfile"):
            mkdir("dst_folder")
            mkdir("source1/source2/")
            with open("source1/source2/file", "w") as f:
                f.write("stuff")

            with open("dst_folder/file", "w") as f:
                f.write("garbage")

            move_file("source1/source2/file", "dst_folder")
            with open("dst_folder/file", "r") as f:
                content = f.read()
                self.assertEqual(content, "stuff")

            self.assertFalse(os.path.exists("source1/source2/file"))

    def test_delete_file(self):
        with TestAreaContext("delete/file"):
            mkdir("pathx")
            with self.assertRaises(IOError):
                delete_file("pathx")

            # deleteFile which does not exist - is silently ignored
            delete_file("does/not/exist")

            with open("file", "w") as f:
                f.write("hei")
            symlink("file", "link")
            self.assertTrue(os.path.islink("link"))

            delete_file("file")
            self.assertFalse(os.path.isfile("file"))
            self.assertTrue(os.path.islink("link"))
            delete_file("link")
            self.assertFalse(os.path.islink("link"))

    def test_delete_directory(self):
        with TestAreaContext("delete/directory"):
            # deleteDriecteory which does not exist - is silently ignored
            delete_directory("does/not/exist")

            with open("file", "w") as f:
                f.write("hei")

            with self.assertRaises(IOError):
                delete_directory("file")

            mkdir("link_target/subpath")
            with open("link_target/link_file", "w") as f:
                f.write("hei")

            mkdir("path/subpath")
            with open("path/file", "w") as f:
                f.write("hei")

            with open("path/subpath/file", "w") as f:
                f.write("hei")

            symlink("../link_target", "path/link")
            delete_directory("path")
            self.assertFalse(os.path.exists("path"))
            self.assertTrue(os.path.exists("link_target/link_file"))

    def test_copy_directory_error(self):
        with self.assertRaises(IOError):
            copy_directory("does/not/exist", "target")

        with TestAreaContext("copy/directory"):
            with open("file", "w") as f:
                f.write("hei")

            with self.assertRaises(IOError):
                copy_directory("hei", "target")

    def test_copy_file(self):
        with TestAreaContext("copy/file"):
            with self.assertRaises(IOError):
                copy_file("does/not/exist", "target")

            mkdir("path")
            with self.assertRaises(IOError):
                copy_file("path", "target")

            with open("file1", "w") as f:
                f.write("hei")

            copy_file("file1", "file2")
            self.assertTrue(os.path.isfile("file2"))

            copy_file("file1", "path")
            self.assertTrue(os.path.isfile("path/file1"))

            copy_file("file1", "path2/file1")
            self.assertTrue(os.path.isfile("path2/file1"))

            with TestAreaContext("copy/file2"):
                mkdir("root/sub/path")

                with open("file", "w") as f:
                    f.write("Hei ...")

                copy_file("file", "root/sub/path/file")
                self.assertTrue(os.path.isfile("root/sub/path/file"))

                with open("file2", "w") as f:
                    f.write("Hei ...")

                with pushd("root/sub/path"):
                    copy_file("../../../file2")
                    self.assertTrue(os.path.isfile("file2"))

    def test_copy_file2(self):
        with TestAreaContext("copy/file2"):
            mkdir("rms/output")

            with open("file.txt", "w") as f:
                f.write("Hei")

            copy_file("file.txt", "rms/output/")
            self.assertTrue(os.path.isfile("rms/output/file.txt"))

    def test_careful_copy_file(self):
        with TestAreaContext("careful/copy/file"):
            with open("file1", "w") as f:
                f.write("hei")
            with open("file2", "w") as f:
                f.write("hallo")

            careful_copy_file("file1", "file2")
            with open("file2", "r") as f:
                self.assertEqual("hallo", f.readline())

            careful_copy_file("file1", "file3")
            self.assertTrue(os.path.isfile("file3"))


if __name__ == "__main__":
    unittest.main()
