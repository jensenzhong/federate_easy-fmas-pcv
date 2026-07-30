import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIRECTORY = REPOSITORY_ROOT / "configs"
PLAINTEXT_API_KEY = re.compile(
    r"\bapi[_-]?key\b[^\r\n]*?\bsk-[A-Za-z0-9_-]{16,}",
    re.IGNORECASE,
)


def test_yaml_configs_do_not_contain_plaintext_api_keys() -> None:
    config_paths = {
        CONFIGS_DIRECTORY / "config.yaml",
        *CONFIGS_DIRECTORY.glob("**/*.yaml"),
    }
    offending_paths = [
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in sorted(config_paths)
        if path.is_file()
        and PLAINTEXT_API_KEY.search(path.read_text(encoding="utf-8"))
    ]

    assert not offending_paths, (
        "Plaintext API keys found in: " + ", ".join(offending_paths)
    )
