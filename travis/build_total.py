#!/usr/bin/env python
from __future__ import print_function

import re
import os
import sys
import subprocess
import shutil
import codecs
import requests
from contextlib import contextmanager

@contextmanager
def pushd(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    cwd0 = os.getcwd()
    os.chdir(path)

    yield

    os.chdir(cwd0)


def print_python_version():
    print(sys.version)

def call(args):
    arg_str = ' '.join(args)
    print('\nCalling %s' % arg_str)
    status = subprocess.call(args)
    if status:
        exit('subprocess.call error:\n\tcode %d\n\targs %s' % (status, arg_str))


def build(source_dir, install_dir, test, c_flags, cxx_flags, test_flags=None):

    cmake_args = ["cmake",
                  source_dir,
                  "-DBUILD_TESTS=ON",
                  "-DENABLE_PYTHON=ON",
                  "-DINSTALL_CWRAP=OFF",
                  "-DBUILD_APPLICATIONS=ON",
                  "-DCMAKE_INSTALL_PREFIX=%s" % install_dir,
                  "-DINSTALL_ERT_LEGACY=ON",
                  "-DCMAKE_PREFIX_PATH=%s" % install_dir,
                  "-DERT_USE_OPENMP=ON",
                  "-DCMAKE_CXX_FLAGS=%s" % cxx_flags,
                  "-DCMAKE_C_FLAGS=%s" % c_flags
                 ]

    build_dir = os.path.join(source_dir, "build")
    with pushd(build_dir):
        call(cmake_args)
        call(["make"])
        call(["make", "install"])

        if test:
            if test_flags is None:
                test_flags = []

            call(["ctest", "--output-on-failure"] + test_flags)
            call(["bin/test_install"])



class PrBuilder(object):

    def __init__(self, argv):
        rep = argv[1]
        self.build_ert = True
        if rep not in ('ecl', 'res', 'ert'):
            raise KeyError("Error: invalid repository type %s." % rep)
        self.repository = rep
        if rep == 'ecl':
            self.rep_name = 'libecl'
            self.build_ert = True
        if rep == 'res':
            self.rep_name = 'libres'
        if rep == 'ert':
            self.rep_name = 'ert'

        self.test_flags = argv[2:]  # argv = [exec, repo, [L|LE, LABEL]]

    def clone_fetch_merge(self):
        self.clone_merge_repository('libecl')
        self.clone_merge_repository('libres')
        if self.build_ert:
            self.clone_merge_repository('ert')

    def clone_merge_repository(self, rep_name):
        if self.rep_name == rep_name:
            return

        call(["git", "clone", "https://github.com/Statoil/%s" % rep_name])

    def compile_and_build(self, basedir):
        install_dir = os.path.join(basedir, "install")
        if not os.path.isdir(install_dir):
            os.makedirs(install_dir)
        self.compile_ecl(basedir, install_dir)

        if self.rep_name == 'libecl' and sys.platform in ('Darwin', 'darwin'):
            return

        self.compile_res(basedir, install_dir)
        if self.build_ert:
            self.compile_ert(basedir, install_dir)


    def compile_ecl(self, basedir, install_dir):
        if self.repository == 'ecl':
            source_dir = basedir
        else:
            source_dir = os.path.join(basedir, "libecl")

        test = (self.repository == 'ecl')
        c_flags = "-Werror=all"
        cxx_flags = "-Werror -Wno-unused-result"
        build(source_dir, install_dir, test, c_flags, cxx_flags, test_flags=self.test_flags)

    def compile_res(self, basedir, install_dir):
        if self.repository == 'res':
            source_dir = basedir
        else:
            source_dir = os.path.join(basedir, "libres")
        test = (self.repository in ('ecl', 'res'))

        # TODO add c_flags = "-Werror=all"
        c_flags = ""
        cxx_flags = "-Werror -Wno-unused-result"
        build(source_dir, install_dir, test, c_flags, cxx_flags, test_flags=self.test_flags)

    def compile_ert(self, basedir, install_dir):
        if self.repository == 'ert':
            source_dir = basedir
        else:
            source_dir = os.path.join(basedir, "ert")
        c_flags = ""
        cxx_flags = "-Werror -Wno-unused-result"
        build(source_dir, install_dir, True, c_flags, cxx_flags, test_flags=self.test_flags)


def main():
    basedir = os.getcwd()
    print('\n===================')
    print(' '.join(sys.argv))
    print('===================\n')
    print_python_version()
    pr_build = PrBuilder(sys.argv)
    pr_build.clone_fetch_merge()
    pr_build.compile_and_build(basedir)
    print(pr_build)

if __name__ == "__main__":
    main()
