"""Settings: environment parsing and the `.env` file the example promises."""

from __future__ import annotations

from shilp.config import Settings, load_dotenv, reset_settings_cache


def test_dotenv_is_read(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "# a comment\n\nSHILP_PORT=9123\nSHILP_ENGINE=fake\nnot a pair\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SHILP_PORT", raising=False)
    monkeypatch.delenv("SHILP_ENGINE", raising=False)
    reset_settings_cache()

    settings = Settings.from_env()
    assert settings.port == 9123
    assert settings.engine == "fake"


def test_real_environment_wins_over_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("SHILP_PORT=9123\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHILP_PORT", "9999")
    reset_settings_cache()

    assert Settings.from_env().port == 9999


def test_missing_dotenv_is_not_an_error(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == 0


def test_quotes_are_stripped(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text('SHILP_ENV_FILE="x"\nHF_TOKEN=\'abc\'\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    reset_settings_cache()
    assert Settings.from_env().hf_token == "abc"


def test_derived_paths_live_under_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SHILP_HOME", str(tmp_path / "home"))
    reset_settings_cache()
    settings = Settings.from_env()
    assert settings.blobs_dir == tmp_path / "home" / "blobs"
    assert settings.db_path == tmp_path / "home" / "logs" / "shilp.db"
