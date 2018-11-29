#!/usr/bin/env python
import os.path
import sys
import subprocess
import shutil
import tempfile
from contextlib import contextmanager

github_repo = "joakim-hove/joakim-hove.github.io"

def call(cmd):
    subprocess.check_call(cmd.split())


@contextmanager
def pushd(path, delete_on_exit=False):
    cwd0 = os.getcwd()
    os.chdir(path)

    yield

    os.chdir(cwd0)
    if delete_on_exit:
        shutil.rmtree(path)


def make_index():
    with open("index.html", "w") as f:
        for e in os.listdir(os.getcwd()):
            if e.startswith("."):
                continue

            if os.path.isdir(e):
                f.write('<a href="{path}/{path}.html"> {path} </a><p>\n'.format(path=e))

            if os.path.isfile(e):
                f.write('<a href="{path}"> {path} </a>'.format(path=e))

    call("git add index.html")



def publish(root, _html_dir, _pdf_file):
    html_dir = os.path.basename(_html_dir)
    pdf_file = os.path.basename(_pdf_file)

    with pushd(tempfile.mkdtemp(), True):
        call("git clone git@github.com:{} docs".format( github_repo ))
        with pushd("docs"):
            call("git rm -fr {}".format(html_dir))
            shutil.copytree(os.path.join(root, _html_dir), html_dir)
            call("git add -A {}".format(html_dir))

            if os.path.isfile(os.path.join(root, _pdf_file)):
                if os.path.isfile(pdf_file):
                    call("git rm {}".format(pdf_file))
                shutil.copy(os.path.join(root, _pdf_file), os.getcwd())
                call("git add {}".format(pdf_file))
            else:
                print("No pdf file specified")

            if not os.path.isfile(".nojekyll"):
                with open(".nojekyll","w") as f:
                    f.write("")
                call("git add .nojekyll")

            make_index()
            try:
                call('git commit -m "Newmanual"')
                call('git push origin master')
            except subprocess.CalledProcessError:
                pass

doc_root = os.path.abspath("docs")
if len(sys.argv) > 1:
    doc_root = os.path.abspath(sys.argv[1])

publish(doc_root, "html/manual", "latex/manual/ERT.pdf")
