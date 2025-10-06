import contextlib
import datetime
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import py
import resfo
from jinja2 import Environment, FileSystemLoader
from numpy import array
from resdata.summary import Summary

from ert.dark_storage.app import app


def source_dir() -> Path:
    current_path = Path(__file__)
    source = current_path.parent.parent
    if not (source / "test-data" / "ert" / "poly_template").exists():
        raise RuntimeError("Cannot find the source folder")
    return source


def write_summary_spec(file, keywords):
    content = [
        ("INTEHEAD", array([1, 100], dtype=np.int32)),
        ("RESTART ", [b"        "] * 8),
        ("DIMENS  ", array([1 + len(keywords), 10, 10, 10, 0, -1], dtype=np.int32)),
        ("KEYWORDS", [f"{x: <8}" for x in ["TIME", *keywords]]),
        ("WGNAMES ", [b":+:+:+:+"] * (len(keywords) + 1)),
        ("NUMS    ", array([-32676] + ([0] * len(keywords)), dtype=np.int32)),
        ("UNITS   ", [f"{x: <8}" for x in ["DAYS"] + ["None"] * len(keywords)]),
        ("STARTDAT", array([1, 1, 2010, 0, 0, 0], dtype=np.int32)),
    ]
    resfo.write(file, content)


def write_summary_data(file, x_size, keywords, update_steps):
    num_keys = len(keywords)

    def content_generator():
        for x in range(x_size):
            yield "SEQHDR  ", array([0], dtype=np.int32)
            for m in range(update_steps):
                step = x * update_steps + m
                day = float(step + 1)
                values = [5.0] * num_keys
                yield "MINISTEP", array([step], dtype=np.int32)
                yield "PARAMS  ", array([day, *values], dtype=np.float32)

    resfo.write(file, content_generator())


def render_template(folder, template, target, **kwargs):
    output = template.render(kwargs)
    file = folder / target
    Path(file).write_text(output, encoding="utf-8")


def make_poly_example(folder, source, **kwargs):
    folder /= "poly"
    summary_count = kwargs["summary_data_count"]
    gen_obs_count = kwargs["gen_obs_count"]
    summary_data_entries = kwargs["summary_data_entries"]
    update_steps = kwargs["update_steps"]
    parameter_count = kwargs["parameter_count"]
    file_loader = FileSystemLoader(str(folder))  # directory of template file
    env = Environment(loader=file_loader)
    shutil.copytree(source, folder)
    for param_num in range(parameter_count):
        render_template(
            folder,
            env.get_template("coeff_priors.j2"),
            f"coeff_priors_{param_num}",
            param_num=param_num,
            **kwargs,
        )
        render_template(
            folder,
            env.get_template("coeff.tmpl.j2"),
            "coeff.tmpl",
            param_num=param_num,
            **kwargs,
        )
    render_template(
        folder, env.get_template("observations.j2"), "observations", **kwargs
    )
    render_template(folder, env.get_template("poly.ert.j2"), "poly.ert", **kwargs)
    render_template(
        folder, env.get_template("poly_eval.py.j2"), "poly_eval.py", **kwargs
    )
    os.chmod(folder / "poly_eval.py", 0o775)
    for r in range(gen_obs_count):
        render_template(
            folder,
            env.get_template("poly_obs_data.txt.j2"),
            f"poly_obs_data_{r}.txt",
            **kwargs,
        )

    if not os.path.exists(folder / "refcase"):
        os.mkdir(folder / "refcase")

    use_resfo = True

    if use_resfo:
        keywords = [f"PSUM{s}" for s in range(summary_count)]
        write_summary_spec(str(folder) + "/refcase/REFCASE.SMSPEC", keywords)
        write_summary_data(
            str(folder) + "/refcase/REFCASE.UNSMRY",
            summary_data_entries,
            keywords,
            update_steps,
        )
    else:
        summary = Summary.writer(
            str(folder) + "/refcase/REFCASE", datetime.datetime(2010, 1, 1), 10, 10, 10
        )
        for s in range(summary_count):
            summary.add_variable(f"PSUM{s}")
            render_template(
                folder,
                env.get_template("poly_obs_data.txt.j2"),
                f"poly_sum_obs_data_{s}.txt",
                **kwargs,
            )

        for x in range(summary_data_entries * update_steps):
            t_step = summary.add_t_step(x // update_steps + 1, sim_days=x + 1)
            for s in range(summary_count):
                t_step[f"PSUM{s}"] = 5.0

        if summary_count > 0:
            summary.fwrite()

    return folder


def make_poly_template(folder, source_folder, **kwargs):
    return make_poly_example(
        folder, source_folder / "test-data" / "ert" / "poly_template", **kwargs
    )


@contextlib.contextmanager
def dark_storage_app(monkeypatch):
    monkeypatch.setenv("ERT_STORAGE_NO_TOKEN", "yup")
    monkeypatch.setenv("ERT_STORAGE_RES_CONFIG", "poly.ert")

    yield app


if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder = py.path.local(sys.argv[1])
        if folder.exists():
            folder.remove()
    else:
        folder = py.path.local(tempfile.mkdtemp())

    source_dir = source_dir()

    make_poly_example(
        folder=folder,
        source=source_dir / "test-data/ert/poly_template",
        gen_data_count=3400,
        gen_data_entries=150,
        summary_data_entries=100,
        reals=200,
        summary_data_count=4000,
        sum_obs_count=450,
        gen_obs_count=34,
        sum_obs_every=10,
        gen_obs_every=1,
        parameter_entries=10,
        parameter_count=10,
        update_steps=1,
    )
    print(folder)
