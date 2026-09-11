"""Tiny helper for assembling notebooks from (kind, source) cell lists."""
from __future__ import annotations

import nbformat as nbf


class NB:
    def __init__(self, title: str):
        self.cells: list = []

    def md(self, text: str):
        self.cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))
        return self

    def code(self, text: str):
        self.cells.append(nbf.v4.new_code_cell(text.strip("\n")))
        return self

    def write(self, path: str):
        nb = nbf.v4.new_notebook()
        nb.cells = self.cells
        nb.metadata = {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        }
        with open(path, "w") as f:
            nbf.write(nb, f)
        print(f"wrote {path} ({len(self.cells)} cells)")
