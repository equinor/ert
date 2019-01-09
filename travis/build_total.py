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

GITHUB_ROT13_API_TOKEN = "665o0ropn09p6r7412n91n3s608pq579n3s4p3o0"

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
                  "-DCMAKE_CXX_FLAGS=-Werror -Wno-unused-result",
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
        self.github_api_token = codecs.encode(GITHUB_ROT13_API_TOKEN, 'rot13')
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

        self.pr_map = {}
        self.access_pr()
        self.test_flags = argv[2:]  # argv = [exec, repo, [L|LE, LABEL]]


    def __str__(self):
        ret_str = "Settings: "
        ret_str += "\nRepository type: %s" % self.repository
        ret_str += "\nECL pr number:   %s" % self.pr_map.get('libecl')
        ret_str += "\nRES pr number:   %s" % self.pr_map.get('libres')
        ret_str += "\nERT pr number:   %s" % self.pr_map.get('ert')
        return ret_str




    def parse_pr_description(self):
        ecl_word = "Statoil/libecl#(\\d+)"
        res_word = "Statoil/libres#(\\d+)"
        ert_word = "Statoil/ert#(\\d+)"
        desc = self.pr_description

        match = re.search(ecl_word, desc, re.MULTILINE)
        if match:
            self.pr_map['libecl'] = int(match.group(1))
        match = re.search(res_word, desc, re.MULTILINE)
        if match:
            self.pr_map['libres'] = int(match.group(1))
        match = re.search(ert_word, desc, re.MULTILINE)
        if match:
            self.pr_map['ert'] = int(match.group(1))

    def check_pr_num_consistency(self):
        if self.repository == "ecl":
            pr_num = self.pr_map.get('libecl')
        elif self.repository == "res":
            pr_num = self.pr_map.get('libres')
        elif self.repository == "ert":
            pr_num = self.pr_map.get('ert')

        if pr_num is not None:
            if pr_num != self.pr_number:
                sys.exit("Error: The line rep=%d does not match pull request %d" %
                         (pr_num, self.pr_number))

    def pr_is_open(self, rep_name, pr_num=None):
        if pr_num is None:
            return
        url = "https://api.github.com/repos/Statoil/%s/pulls/%d" % (rep_name, pr_num)
        github_api_token = os.getenv("GITHUB_API_TOKEN")
        response = requests.get(url, {"access_token" : self.github_api_token})

        content = response.json()
        state = content["state"]
        return state == "open"

    def access_pr(self):
        if "TRAVIS_PULL_REQUEST" not in os.environ:
            return

        pr_number_string = os.getenv("TRAVIS_PULL_REQUEST")
        if pr_number_string == "false":
            return

        self.pr_number = int(pr_number_string)
        url = "https://api.github.com/repos/Statoil/%s/pulls/%d" % (self.rep_name, self.pr_number)
        print("Accessing: %s" % url)

        response = requests.get(url, {"access_token" : self.github_api_token})

        if response.status_code != 200:
            sys.exit("HTTP GET from GitHub failed: %s" % response.text)

        content = response.json()
        self.pr_description = content["body"]
        print("PULL REQUEST: %d\n%s" % (self.pr_number, self.pr_description))
        self.parse_pr_description()
        self.check_pr_num_consistency()

        # Removing referred-to closed PRs
        for rep in ('libecl', 'libres', 'ert'):
            if not self.pr_is_open(rep, self.pr_map.get(rep)):
                print('Ignoring referred-to closed %s pr %s' % (rep, self.pr_map.get(rep)))
                self.pr_map[rep] = None


    def clone_fetch_merge(self, basedir):
        self.clone_merge_repository('libecl', self.pr_map.get('libecl'), basedir)
        self.clone_merge_repository('libres', self.pr_map.get('libres'), basedir)
        if self.build_ert:
            self.clone_merge_repository('ert', self.pr_map.get('ert'), basedir)

    def clone_merge_repository(self, rep_name, pr_num, basedir):
        if self.rep_name == rep_name:
            return

        call(["git", "clone", "https://github.com/Statoil/%s" % rep_name])
        if pr_num is None:
            return
        rep_path = os.path.join(basedir, rep_name)
        cwd = os.getcwd()
        os.chdir(rep_path)
        call(["git", "config", "user.email", "you@example.com"])
        call(["git", "config", "user.name", "Your Name"])
        path = "refs/pull/%d/head:%d" % (pr_num, pr_num)
        call(["git", "fetch", "-f", "origin", path])
        call(["git", "merge", "%d" % pr_num, '-m"A MESSAGE"'])
        os.chdir(cwd)

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
        cxx_flags = ""
        build(source_dir, install_dir, test, c_flags, cxx_flags, test_flags=self.test_flags)

    def compile_ert(self, basedir, install_dir):
        if self.repository == 'ert':
            source_dir = basedir
        else:
            source_dir = os.path.join(basedir, "ert")
        c_flags = ""
        cxx_flags = ""
        build(source_dir, install_dir, True, c_flags, cxx_flags, test_flags=self.test_flags)


def main():
    basedir = os.getcwd()
    print('\n===================')
    print(' '.join(sys.argv))
    print('===================\n')
    print_python_version()
    pr_build = PrBuilder(sys.argv)
    pr_build.clone_fetch_merge(basedir)
    pr_build.compile_and_build(basedir)
    print(pr_build)

if __name__ == "__main__":
    main()
