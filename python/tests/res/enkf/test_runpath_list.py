import unittest
import os.path

from ecl.util.test import TestAreaContext
from tests import ResTest
from res.test import ErtTestContext

from res.enkf import RunpathList, RunpathNode, ErtRunContext
from res.enkf.enums import EnkfInitModeEnum, EnkfRunType
from ecl.util.util import BoolVector
from res.util.substitution_list import SubstitutionList


def path(idx):
    return "path_%d" % idx


def base(idx):
    return "base_%d" % idx


class RunpathListTest(ResTest):
    def test_load(self):
        with TestAreaContext("runpath_list"):
            with self.assertRaises(IOError):
                rp = RunpathList("")

            node0 = RunpathNode(0, 0, "path0", "base0")
            node1 = RunpathNode(1, 1, "path1", "base1")

            runpath_list = RunpathList("filex.list")
            with self.assertRaises(IOError):
                runpath_list.load()

            with open("invalid_file", "w") as f:
                f.write("X X X X\n")

            rp = RunpathList("invalid_file")
            with self.assertRaises(IOError):
                rp.load()

            rp = RunpathList("list.txt")
            rp.add(node0.realization, node0.iteration, node0.runpath, node0.basename)
            rp.add(node1.realization, node1.iteration, node1.runpath, node1.basename)
            rp.export()
            self.assertTrue(os.path.isfile("list.txt"))

            rp2 = RunpathList("list.txt")
            rp2.load()
            self.assertEqual(len(rp2), 2)
            self.assertEqual(rp2[0], node0)
            self.assertEqual(rp2[1], node1)

    def test_runpath_list(self):
        runpath_list = RunpathList("file")

        self.assertEqual(len(runpath_list), 0)

        test_runpath_nodes = [
            RunpathNode(0, 0, "runpath0", "basename0"),
            RunpathNode(1, 0, "runpath1", "basename0"),
        ]

        runpath_node = test_runpath_nodes[0]
        runpath_list.add(
            runpath_node.realization,
            runpath_node.iteration,
            runpath_node.runpath,
            runpath_node.basename,
        )

        self.assertEqual(len(runpath_list), 1)
        self.assertEqual(runpath_list[0], test_runpath_nodes[0])

        runpath_node = test_runpath_nodes[1]
        runpath_list.add(
            runpath_node.realization,
            runpath_node.iteration,
            runpath_node.runpath,
            runpath_node.basename,
        )

        self.assertEqual(len(runpath_list), 2)
        self.assertEqual(runpath_list[1], test_runpath_nodes[1])

        for index, runpath_node in enumerate(runpath_list):
            self.assertEqual(runpath_node, test_runpath_nodes[index])

        runpath_list.clear()

        self.assertEqual(len(runpath_list), 0)

    def test_collection(self):
        """Testing len, adding, getting (idx and slice), printing, clearing."""
        with TestAreaContext("runpath_list_collection"):
            runpath_list = RunpathList("EXPORT.txt")
            runpath_list.add(3, 1, path(3), base(3))
            runpath_list.add(1, 1, path(1), base(1))
            runpath_list.add(2, 1, path(2), base(2))
            runpath_list.add(0, 0, path(0), base(0))
            runpath_list.add(3, 0, path(3), base(3))
            runpath_list.add(1, 0, path(1), base(1))
            runpath_list.add(2, 0, path(2), base(2))
            runpath_list.add(0, 1, path(0), base(0))

            self.assertEqual(8, len(runpath_list))
            pfx = "RunpathList(size"  # the __repr__ function
            self.assertEqual(pfx, repr(runpath_list)[: len(pfx)])
            node2 = RunpathNode(2, 1, path(2), base(2))
            self.assertEqual(node2, runpath_list[2])

            node3 = RunpathNode(0, 0, path(0), base(0))
            node4 = RunpathNode(3, 0, path(3), base(3))
            node5 = RunpathNode(1, 0, path(1), base(1))
            node6 = RunpathNode(2, 0, path(2), base(2))
            nodeslice = [node3, node4, node5, node6]
            self.assertEqual(nodeslice, runpath_list[3:7])
            self.assertEqual(node6, runpath_list[-2])
            with self.assertRaises(TypeError):
                runpath_list["key"]
            with self.assertRaises(IndexError):
                runpath_list[12]

            runpath_list.clear()
            self.assertEqual(0, len(runpath_list))
            with self.assertRaises(IndexError):
                runpath_list[0]
            self.assertEqual("EXPORT.txt", runpath_list.getExportFile())

    def test_sorted_export(self):
        with TestAreaContext("runpath_list_sorted"):
            runpath_list = RunpathList("EXPORT.txt")
            runpath_list.add(3, 1, "path", "base")
            runpath_list.add(1, 1, "path", "base")
            runpath_list.add(2, 1, "path", "base")
            runpath_list.add(0, 0, "path", "base")

            runpath_list.add(3, 0, "path", "base")
            runpath_list.add(1, 0, "path", "base")
            runpath_list.add(2, 0, "path", "base")
            runpath_list.add(0, 1, "path", "base")

            runpath_list.export()

            path_list = []
            with open("EXPORT.txt") as f:
                for line in f.readlines():
                    tmp = line.split()
                    iens = int(tmp[0])
                    iteration = int(tmp[3])

                    path_list.append((iens, iteration))

            for iens in range(4):
                t0 = path_list[iens]
                t4 = path_list[iens + 4]
                self.assertEqual(t0[0], iens)
                self.assertEqual(t4[0], iens)

                self.assertEqual(t0[1], 0)
                self.assertEqual(t4[1], 1)
