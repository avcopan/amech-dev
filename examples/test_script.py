#!/usr/bin/env python
import re
import textwrap
from pathlib import Path

import more_itertools as mit
import pyparsing as pp
from pyparsing import common as ppc

COMMENT_REGEX = re.compile(r"#.*$", flags=re.M)


def run_test(source_dir, nodes):
    source_dir = Path(source_dir)
    test_dir = Path(".test") / source_dir.name
    test_dir.mkdir(parents=True, exist_ok=True)
    test_dir = test_dir.resolve()

    file_dct = read_input_files(source_dir)
    run_dct = parse_run_dat(file_dct.get("run.dat"))
    # 0. Insert absolute paths for the run and save directories
    run_dct["input"] = absolute_path_input_block(run_dct.get("input"), source_dir)
    # 1. Extract species tasks and indices
    spc_tasks = tasks_from_elstruct_block(run_dct.get("els"), "spc")
    spc_idxs = indices_from_species_block(run_dct.get("spc"))
    # 2. Run the tasks for each species
    #       a. Outer loop: Run tasks sequentially
    for num, task in enumerate(spc_tasks[:1]):
        task_run_dct = run_dct.copy()
        task_run_dct["els"] = task
        #   b. Inner loop: Run species in parallel
        task_name = task.split()[1]
        spc_run_dirs = []
        for spc_idx in spc_idxs:
            spc_run_dct = {**task_run_dct, "spc": f"{spc_idx}"}
            spc_file_dct = {**file_dct, "run.dat": form_run_dat(spc_run_dct)}
            spc_run_dir = Path(test_dir) / f"{num}_spc_{task_name}" / f"{spc_idx:02d}"
            spc_run_dir.mkdir(parents=True, exist_ok=True)
            write_input_files(spc_run_dir, spc_file_dct)
            spc_run_dirs.append(str(spc_run_dir))
        print(spc_run_dirs)

    import subprocess

    subprocess.run(
        ["pixi", "run", "test", "20", "9", ",".join(spc_run_dirs), ",".join(nodes)]
    )


# Helpers
# def run_dat_from_blocks
def read_input_files(run_dir: str | Path) -> dict[str, str]:
    inp_dir = Path(run_dir) / "inp"
    return {
        "run.dat": (inp_dir / "run.dat").read_text(),
        "theory.dat": (inp_dir / "theory.dat").read_text(),
        "models.dat": (inp_dir / "models.dat").read_text(),
        "mechanism.dat": (inp_dir / "mechanism.dat").read_text(),
        "species.csv": (inp_dir / "species.csv").read_text(),
    }


def write_input_files(run_dir: str | Path, file_dct: dict[str, str]) -> None:
    inp_dir = Path(run_dir) / "inp"
    inp_dir.mkdir(exist_ok=True)
    for name, contents in file_dct.items():
        (inp_dir / name).write_text(contents)


def form_run_dat(run_dct: dict[str, str]) -> str:
    keys = ["input", "spc", "pes", "els", "thermo", "kin"]
    run_dat = ""
    for key in keys:
        run_dat += f"{key}\n{format_block(run_dct.get(key))}\nend {key}\n\n"
    return run_dat


def parse_run_dat(run_dat: str) -> dict[str, str]:
    def _parse_block(run_dat, keyword):
        start = pp.Keyword(keyword) + pp.LineEnd()
        end = pp.Keyword("end") + pp.Keyword(keyword)
        expr = pp.Suppress(... + start) + pp.SkipTo(end)("content")
        content = expr.parseString(run_dat).get("content")
        return format_block(content)

    return {
        "input": _parse_block(run_dat, "input"),
        "pes": _parse_block(run_dat, "pes"),
        "spc": _parse_block(run_dat, "spc"),
        "els": _parse_block(run_dat, "els"),
        "thermo": _parse_block(run_dat, "thermo"),
        "kin": _parse_block(run_dat, "kin"),
    }


def absolute_path_input_block(inp_block: str, dir: str) -> str:
    def _resolve_path(path: str) -> str:
        if Path(path).is_absolute():
            return path
        return str((Path(dir) / path).resolve())

    inp_block = without_comments(inp_block)
    word = pp.Word(pp.printables, exclude_chars="=")
    line = pp.Group(word("key") + pp.Literal("=") + word("val"))
    expr = pp.DelimitedList(line, delim=pp.LineEnd())
    path_dct = {r.get("key"): r.get("val") for r in expr.parseString(inp_block)}

    lines = [f"{key} = {_resolve_path(val)}" for key, val in path_dct.items()]
    return "\n".join(lines)


def indices_from_species_block(spc_block: str) -> list[int]:
    dash = pp.Suppress(pp.Literal("-"))
    entry = ppc.integer ^ pp.Group(ppc.integer + dash + ppc.integer)
    expr = pp.DelimitedList(entry, delim=pp.LineEnd())
    idxs = []
    for res in expr.parseString(spc_block).as_list():
        if isinstance(res, int):
            idxs.append(res)
        else:
            start, stop = res
            idxs.extend(range(start, stop + 1))
    return list(mit.unique_everseen(idxs))


def tasks_from_elstruct_block(els_block: str, key: str) -> list[str]:
    lines = [line.strip() for line in els_block.splitlines()]
    return [line for line in lines if line.startswith(key)]


def format_block(inp: str) -> str:
    """Format a block with nice indentation

    :param inp: A multiline string to be formatted
    :return: The formatted string
    """
    inp = re.sub(r"^\ *", "", inp, flags=re.MULTILINE)
    return textwrap.indent(inp, "    ")


def without_comments(inp: str) -> str:
    """Get a CHEMKIN string or substring with comments removed.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The string, without comments
    """
    return re.sub(COMMENT_REGEX, "", inp)


if __name__ == "__main__":
    run_test(
        source_dir="propyl",
        nodes=["csed-0009", "csed-0010", "csed-0011"],
    )
