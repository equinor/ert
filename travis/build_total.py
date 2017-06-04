#!/usr/bin/env python
import re
import requests
import json
import os
import sys
import subprocess
import shutil



class build_class():

    def __init__(self, rep, basedir):
        if rep in ('ecl', 'res', 'ert'):
            self.repository = rep
            if (rep == 'ecl'):
                self.rep_name = 'libecl'
            if (rep == 'res'):
                self.rep_name = 'libres'
            if (rep == 'ert'):
                self.rep_name = 'ert'
            self.ecl_pr_num = -1
            self.res_pr_num = -1
            self.ert_pr_num = -1
            self.access_pr()
            self.clone_fetch_merge(basedir)
            #self.compile_and_build(basedir)
           
        else:
            raise Exception("Error: invalid repository type.")
    
    def display(self):
        print "Settings: "
        print "Repository type: %s" % self.repository
        print "ECL pr number: %d" % self.ecl_pr_num
        print "RES pr number: %d" % self.res_pr_num
        print "ERT pr number: %d" % self.ert_pr_num

    def parse_pr_description(self):
        ecl_word = "^ecl=(\d+)"
        res_word = "^res=(\d+)"
        ert_word = "^ert=(\d+)"
        st = self.pr_description
        m = re.search(ecl_word, st, re.MULTILINE)
        if m:
            self.ecl_pr_num = int(m.group(1))       
        m = re.search(res_word, st, re.MULTILINE)
        if m:
            self.res_pr_num = int(m.group(1))
        m = re.search(ert_word, st, re.MULTILINE)
        if m:
            self.ert_pr_num = int(m.group(1))
   
    def check_pr_num_consistency(self):
        if (self.repository == "ecl"):
            pr_num = self.ecl_pr_num
        if (self.repository == "res"):
            pr_num = self.res_pr_num
        if (self.repository == "ert"):
            pr_num = self.ert_pr_num
        if (pr_num >= 0):
            if (pr_num != self.pr_number):
                sys.exit("Error: The line rep=%d does not match pull request %d" % (pr_num, self.pr_number))
                
        else:
            if (self.repository == "ecl"):
                self.ecl_pr_num = self.pr_number
            if (self.repository == "res"):
                self.res_pr_num = self.pr_number
            if (self.repository == "ert"):
                self.ert_pr_num = self.pr_number

    def assert_open_pr_status(self, rep_name, pr_num):
        if (pr_num >= 0):
            url = "https://api.github.com/repos/Statoil/%s/pulls/%d" % (rep_name, pr_num)
            if "GITHUB_API_TOKEN" in os.environ:
                github_api_token = os.getenv("GITHUB_API_TOKEN")
                response = requests.get( url , {"access_token" : github_api_token})
            else:
                response = requests.get( url )
            content = json.loads( response.content )
            state = content["state"]
            assert(state == "open")
            
    def access_pr(self): 
        pr_number_string = os.getenv("TRAVIS_PULL_REQUEST")
        self.pr_number = int(pr_number_string)
        url = "https://api.github.com/repos/Statoil/%s/pulls/%d" % (self.rep_name, self.pr_number)
        print "Accessing: %s" % url

        if "GITHUB_API_TOKEN" in os.environ: 
            github_api_token = os.getenv("GITHUB_API_TOKEN")
            response = requests.get( url , {"access_token" : github_api_token})
        else:
            response = requests.get( url )
    
        content = json.loads( response.content )
        self.pr_description = content["body"]
        print "PULL REQUEST: %d\n%s" % (self.pr_number, self.pr_description)
        self.parse_pr_description();
        self.check_pr_num_consistency();
        self.assert_open_pr_status('libecl', self.ecl_pr_num)
        self.assert_open_pr_status('libres', self.res_pr_num)
        self.assert_open_pr_status('ert'   , self.ert_pr_num)

    def clone_fetch_merge(self, basedir):      
        self.clone_merge_repository('libecl', self.ecl_pr_num, basedir)
        self.clone_merge_repository('libres', self.res_pr_num, basedir)
        self.clone_merge_repository('ert'   , self.ert_pr_num, basedir)

    def clone_merge_repository(self, rep_name, pr_num, basedir):
        if (self.rep_name != rep_name):
            subprocess.check_call(["git", "clone", "https://github.com/Statoil/%s" % rep_name])
            if (pr_num >= 0):
                rep_path = os.path.join(basedir, rep_name)
                cwd = os.getcwd()
                os.chdir(rep_path)
                subprocess.check_call(["git", "config", "user.email", "you@example.com"])
                subprocess.check_call(["git", "config", "user.name", "Your Name"])
                s = "refs/pull/%d/head:%d" % (pr_num, pr_num)
                subprocess.check_call(["git", "fetch", "-f", "origin", s])
                subprocess.check_call(["git", "merge", "%d" % pr_num, '-m"A MESSAGE"'])
                os.chdir(cwd) 
            
    def compile_and_build(self, basedir):
        install_dir = os.path.join(basedir, "install")
        if not os.path.isdir(install_dir):
            os.makedirs(install_dir)
        self.compile_ecl(basedir, install_dir)
        self.compile_res(basedir, install_dir)
        self.compile_ert(basedir, install_dir)

    def build(self, source_dir, install_dir, test):
        build_dir = os.path.join(source_dir, "build")
        if not os.path.isdir(build_dir):
            os.makedirs(build_dir)
        cmake_args = ["cmake", source_dir, "-DBUILD_TESTS=ON", "-DBUILD_PYTHON=ON", "-DERT_BUILD_CXX=ON", "-DBUILD_APPLICATIONS=ON", "-DCMAKE_INSTALL_PREFIX=%s" % install_dir, 
                      "-DINSTALL_ERT_LEGACY=ON", "-DCMAKE_PREFIX_PATH=%s" % install_dir, "-DCMAKE_MODULE_PATH=%s/share/cmake/Modules" % install_dir]
        cwd = os.getcwd()
        os.chdir(build_dir)
        subprocess.check_call(cmake_args)
        subprocess.check_call(["make"])
        if test:
            subprocess.check_call(["ctest"])
        subprocess.check_call(["make" , "install"])
        subprocess.check_call(["bin/test_install"])
        os.chdir(cwd) 

    def compile_ecl(self, basedir, install_dir):
        if (self.repository == 'ecl'):
            source_dir = basedir
        else:
            source_dir = os.path.join(basedir, "libecl")
        if (self.repository == 'ecl'):
            test = True
        else:
            test = False
        self.build(source_dir, install_dir, test)

    def compile_res(self, basedir, install_dir):
        if (self.repository == 'res'):
            source_dir = basedir
        else:
            source_dir = os.path.join(basedir, "libres")
        if (self.repository == 'ecl' or self.repository == 'res'):
            test = True
        else:
            test = False
        self.build(source_dir, install_dir, test)

    def compile_ert(self, basedir, install_dir):
        if (self.repository == 'ert'):
            source_dir = basedir
        else:
            source_dir = os.path.join(basedir, "ert")
        self.build(source_dir, install_dir, True)
        

def main():
    basedir = os.getcwd()
    pr_build = build_class(sys.argv[1], basedir)
    pr_build.display()
    sys.exit(0)
   
        
if __name__ == "__main__":
    main( )
