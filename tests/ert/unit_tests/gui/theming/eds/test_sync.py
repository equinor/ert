import io
import json
import runpy
import sys
import urllib.request
from contextlib import contextmanager

from ert.gui.theming.eds import data, sync
from ert.gui.theming.theme import ColorScheme

_FAKE_SCALE = {"accent-11": "#0084c4"}
_FAKE_SEMANTIC = {"bg-accent-fill-muted-default": "#e6f4fb"}


def _fake_urlopen_factory():
    def fake_urlopen(url, timeout=None):
        payload = _FAKE_SEMANTIC if url.endswith("-semantic.json") else _FAKE_SCALE

        @contextmanager
        def _cm():
            yield io.BytesIO(json.dumps(payload).encode("utf-8"))

        return _cm()

    return fake_urlopen


def _redirect_bundles_to(tmp_path, monkeypatch):
    paths = {scheme: tmp_path / f"{scheme.value}.json" for scheme in ColorScheme}
    # ``sync`` binds ``bundled_path`` at import via ``from .data import ...``; patch
    # both so direct calls and a fresh ``runpy`` import target the temp directory.
    monkeypatch.setattr(sync, "bundled_path", lambda scheme: paths[scheme])
    monkeypatch.setattr(data, "bundled_path", lambda scheme: paths[scheme])
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen_factory())
    return paths


def test_that_sync_writes_one_bundle_file_per_color_scheme(tmp_path, monkeypatch):
    paths = _redirect_bundles_to(tmp_path, monkeypatch)

    written = sync.sync()

    assert set(written) == set(paths.values())
    for path in paths.values():
        assert path.is_file()


def test_that_sync_merges_scale_and_semantic_layers_into_each_bundle(
    tmp_path, monkeypatch
):
    paths = _redirect_bundles_to(tmp_path, monkeypatch)

    sync.sync()

    bundle = json.loads(paths[ColorScheme.LIGHT].read_text(encoding="utf-8"))
    assert bundle == {"scale": _FAKE_SCALE, "semantic": _FAKE_SEMANTIC}


def test_that_running_the_sync_module_prints_each_written_path(
    tmp_path, monkeypatch, capsys
):
    paths = _redirect_bundles_to(tmp_path, monkeypatch)
    # Drop the cached import so ``runpy`` re-executes the module cleanly as a
    # script instead of warning about it already being in ``sys.modules``.
    monkeypatch.delitem(sys.modules, "ert.gui.theming.eds.sync", raising=False)

    runpy.run_module("ert.gui.theming.eds.sync", run_name="__main__")

    out = capsys.readouterr().out
    for path in paths.values():
        assert f"wrote {path}" in out
