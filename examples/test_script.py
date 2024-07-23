#!/usr/bin/env python
import re
import textwrap
from pathlib import Path

import more_itertools as mit
import pyparsing as pp
from pyparsing import common as ppc

COMMENT_REGEX = re.compile(r"#.*$", flags=re.M)


def blocks_from_run_dat(run_dat):
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
    return format_block("\n".join(lines))


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


def run_test(test_dir):
    run_dct = blocks_from_run_dat(open(Path(test_dir) / "inp" / "run.dat").read())
    spc_idxs = indices_from_species_block(run_dct.get("spc"))
    print(spc_idxs)
    spc_tasks = tasks_from_elstruct_block(run_dct.get("els"), "spc")
    for task in spc_tasks:
        print(task)
    ts_tasks = tasks_from_elstruct_block(run_dct.get("els"), "ts")
    for task in ts_tasks:
        print(task)
    inp_block = absolute_path_input_block(run_dct.get("input"), test_dir)
    print(inp_block)


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
    run_test("propyl")
