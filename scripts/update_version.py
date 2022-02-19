import os
from pathlib import Path
import typer
import re
import toml


def main(release_name: str):
    release_name = release_name.replace("-create-release", "")

    # Update pyproject.toml
    pyproject_path = Path("pyproject.toml")
    pyproject = toml.load(pyproject_path)
    pyproject["tool"]["poetry"]["version"] = release_name
    toml.dump(pyproject, pyproject_path.open("w"))

    # Update __init__.py
    init_path = Path("treeo/__init__.py")
    init_text = init_path.read_text()
    init_text = re.sub(
        r'__version__ = "(.*?)"', f'__version__ = "{release_name}"', init_text
    )
    init_path.write_text(init_text)
