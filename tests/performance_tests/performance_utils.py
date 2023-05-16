import contextlib
import datetime
import os
import shutil
import sys
import tempfile

import ecl_data_io as eclio
import numpy
import py
from ecl.summary import EclSum
from jinja2 import Environment, FileSystemLoader
from numpy import array

from ert.dark_storage import enkf


def write_summary_spec(file, keywords):
    content = [
        ("INTEHEAD", [1, 100]),
        ("RESTART ", [b"        "] * 8),
        ("DIMENS  ", [1 + len(keywords), 10, 10, 10, 0, -1]),
        ("KEYWORDS", [f"{x: <8}" for x in ["TIME"] + keywords]),
        ("WGNAMES ", [b":+:+:+:+"] * (len(keywords) + 1)),
        ("NUMS    ", [-32676] + ([0] * len(keywords))),
        ("UNITS   ", [f"{x: <8}" for x in ["DAYS"] + ["None"] * len(keywords)]),
        ("STARTDAT", [1, 1, 2010, 0, 0, 0]),
    ]
    eclio.write(file, content)


def write_summary_data(file, x_size, keywords, update_steps):
    num_keys = len(keywords)

    def content_generator():
        for x in range(x_size):
            yield "SEQHDR  ", [0]
            for m in range(update_steps):
                step = x * update_steps + m
                day = float(step + 1)
                values = [5.0] * num_keys
                yield "MINISTEP", [step]
                yield "PARAMS  ", array([day] + values, dtype=numpy.float32)

    eclio.write(file, content_generator())


def render_template(folder, template, target, **kwargs):
    output = template.render(kwargs)
    file = folder / target
    with open(file, "w", encoding="utf-8") as f:
        f.write(output)


def make_poly_example(folder, source, **kwargs):
    folder = folder / "poly"
    summary_count = kwargs["summary_data_count"]
    gen_obs_count = kwargs["gen_obs_count"]
    summary_data_entries = kwargs["summary_data_entries"]
    update_steps = kwargs["update_steps"]
    file_loader = FileSystemLoader(str(folder))  # directory of template file
    env = Environment(loader=file_loader)
    shutil.copytree(source, folder)

    render_template(
        folder, env.get_template("coeff_priors.j2"), "coeff_priors", **kwargs
    )
    render_template(folder, env.get_template("coeff.tmpl.j2"), "coeff.tmpl", **kwargs)
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

    use_ecl_data_io = True

    if use_ecl_data_io:
        keywords = [f"PSUM{s}" for s in range(summary_count)]
        write_summary_spec(str(folder) + "/refcase/REFCASE.SMSPEC", keywords)
        write_summary_data(
            str(folder) + "/refcase/REFCASE.UNSMRY",
            summary_data_entries,
            keywords,
            update_steps,
        )
    else:
        ecl_sum = EclSum.writer(
            str(folder) + "/refcase/REFCASE", datetime.datetime(2010, 1, 1), 10, 10, 10
        )
        for s in range(summary_count):
            ecl_sum.addVariable(f"PSUM{s}")
            render_template(
                folder,
                env.get_template("poly_obs_data.txt.j2"),
                f"poly_sum_obs_data_{s}.txt",
                **kwargs,
            )

        for x in range(summary_data_entries * update_steps):
            t_step = ecl_sum.addTStep(x // update_steps + 1, sim_days=x + 1)
            for s in range(summary_count):
                t_step[f"PSUM{s}"] = 5.0

        if summary_count > 0:
            ecl_sum.fwrite()

    return folder


def make_poly_template(folder, source_folder, **kwargs):
    return make_poly_example(
        folder, source_folder / "test-data" / "poly_template", **kwargs
    )


def reset_enkf():
    enkf.ids = {}
    enkf._config = None
    enkf._ert = None
    enkf._libres_facade = None


@contextlib.contextmanager
def dark_storage_app(monkeypatch):
    monkeypatch.setenv("ERT_STORAGE_NO_TOKEN", "yup")
    monkeypatch.setenv("ERT_STORAGE_RES_CONFIG", "poly.ert")
    monkeypatch.setenv("ERT_STORAGE_DATABASE_URL", "sqlite://")
    from ert.dark_storage.app import app

    yield app
    reset_enkf()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder = py.path.local(sys.argv[1])
        if folder.exists():
            folder.remove()
    else:
        folder = py.path.local(tempfile.mkdtemp())
    make_poly_example(
        folder,
        "../../test-data/poly_template",
        gen_data_count=34,
        gen_data_entries=15,
        summary_data_entries=100,
        reals=200,
        summary_data_count=4000,
        sum_obs_count=450,
        gen_obs_count=34,
        sum_obs_every=10,
        gen_obs_every=1,
        parameter_entries=10,
        parameter_count=8,
        update_steps=1,
    )
    print(folder)
