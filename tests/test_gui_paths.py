"""Portable-mode rules (Qt-free: the decision is plain filesystem logic)."""

from pathlib import Path

from celclean.gui import paths


def test_settings_file_is_none_when_running_from_source(tmp_path, monkeypatch) -> None:
    # a dev run must not scatter an ini next to the sources
    monkeypatch.delenv(paths.ENV_FORCE, raising=False)
    monkeypatch.setattr(paths, "is_frozen", lambda: False)
    assert paths.settings_file() is None
    assert "注册表" in paths.describe()


def test_force_and_env_both_switch_to_a_file_beside_the_app(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paths, "base_dir", lambda: tmp_path)
    monkeypatch.delenv(paths.ENV_FORCE, raising=False)
    assert paths.settings_file(force=True) == tmp_path / paths.SETTINGS_NAME
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv(paths.ENV_FORCE, value)
        assert paths.settings_file() == tmp_path / paths.SETTINGS_NAME
    monkeypatch.setenv(paths.ENV_FORCE, "0")
    assert paths.settings_file() is None


def test_a_packed_build_is_portable_where_it_can_write(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(paths.ENV_FORCE, raising=False)
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    monkeypatch.setattr(paths, "base_dir", lambda: tmp_path)
    assert paths.settings_file() == tmp_path / paths.SETTINGS_NAME
    assert (tmp_path / paths.SETTINGS_NAME).exists()


def test_a_read_only_location_falls_back_to_the_native_store(tmp_path, monkeypatch) -> None:
    # a file where the directory should be is what an unwritable install directory looks like
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.delenv(paths.ENV_FORCE, raising=False)
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    monkeypatch.setattr(paths, "base_dir", lambda: blocker)
    assert paths.writable(blocker) is False
    assert paths.settings_file() is None
    assert "注册表" in paths.describe()


def test_base_dir_of_a_source_run_is_the_repository_root() -> None:
    root = paths.base_dir()
    assert (root / "pyproject.toml").exists() and (root / "src" / "celclean").is_dir()
