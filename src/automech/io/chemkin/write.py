"""Functions for writing CHEMKIN-formatted files."""

import itertools
from pathlib import Path

import automol
import polars

from ... import _mech, reac_table, schema
from ..._mech import Mechanism
from ...data import reac
from ...schema import Reaction, ReactionRate, ReactionSorted, Species, SpeciesThermo
from ...util import df_
from .read import KeyWord


def mechanism(mech: Mechanism, out: str | Path | None = None) -> str:
    """Write a mechanism to CHEMKIN format.

    :param mech: A mechanism
    :param out: Optionally, write the output to this file path
    :return: The CHEMKIN mechanism as a string
    """
    blocks = [
        elements_block(mech),
        species_block(mech),
        thermo_block(mech),
        reactions_block(mech),
    ]
    mech_str = "\n\n\n".join(b for b in blocks if b is not None)
    if out is not None:
        out: Path = Path(out)
        out.write_text(mech_str)

    return mech_str


def elements_block(mech: Mechanism) -> str:
    """Write the elements block to a string.

    :param mech: A mechanism
    :return: The elements block string
    """
    spc_df = _mech.species(mech)
    fmls = list(map(automol.amchi.formula, spc_df[Species.amchi].to_list()))
    elem_strs = set(itertools.chain(*(f.keys() for f in fmls)))
    elem_strs = automol.form.sorted_symbols(elem_strs)
    return block(KeyWord.ELEMENTS, elem_strs)


def species_block(mech: Mechanism) -> str:
    """Write the species block to a string.

    :param mech: A mechanism
    :return: The species block string
    """
    spc_df = _mech.species(mech)
    name_width = 1 + spc_df[Species.name].str.len_chars().max()
    smi_width = 1 + spc_df[Species.smiles].str.len_chars().max()
    spc_strs = [
        f"{n:<{name_width}} ! SMILES: {s:<{smi_width}} AMChI: {c}"
        for n, s, c in spc_df.select(Species.name, Species.smiles, Species.amchi).rows()
    ]
    return block(KeyWord.SPECIES, spc_strs)


def thermo_block(mech: Mechanism) -> str:
    """Write the thermo block to a string.

    :param mech: A mechanism
    :return: The thermo block string
    """
    spc_df = _mech.species(mech)
    if SpeciesThermo.thermo_string not in spc_df:
        return None

    # Generate the thermo strings
    therm_strs = spc_df.select(
        polars.concat_str(
            polars.col(Species.name).str.pad_end(24),
            polars.col(SpeciesThermo.thermo_string),
        )
    ).to_series()

    # Generate the header
    therm_temps = _mech.thermo_temperatures(mech)
    if therm_temps is None:
        header = None
    else:
        therm_temps_str = "  ".join(f"{t:.3f}" for t in therm_temps)
        header = f"ALL\n    {therm_temps_str}"

    return block(KeyWord.THERM, therm_strs, header=header)


def reactions_block(mech: Mechanism, frame: bool = True) -> str:
    """Write the reactions block to a string.

    :param mech: A mechanism
    :param frame: Whether to frame the block with its header and footer
    :return: The reactions block string
    """
    # Generate reaction objects
    # (Eventually, we should have a function like _mech.with_reaction_objects(mech))
    rxn_df = reac_table.with_rates(_mech.reactions(mech))

    # Generate the header
    rate_units = _mech.rate_units(mech)
    if rate_units is None:
        header = None
    else:
        e_unit, a_unit = rate_units
        header = f"   {e_unit}   {a_unit}"

    if rxn_df.is_empty():
        return block(KeyWord.REACTIONS, "", header=header, frame=frame)

    # Format rate data
    cols = [
        Reaction.reactants,
        Reaction.products,
        ReactionRate.rate,
        ReactionRate.colliders,
    ]
    rxn_df = df_.map_(rxn_df, cols, "obj", reac.from_data, dtype_=object)

    # Determine the max equation width for formatting
    rxn_df = df_.map_(rxn_df, "obj", "ck_eq", reac.chemkin_equation)
    eq_width = 10 + rxn_df["ck_eq"].str.len_chars().max()

    # Detect duplicates
    rxn_df = df_.map_(rxn_df, "obj", "dup_key", reac.chemkin_duplicate_key)
    rxn_df = rxn_df.with_columns(polars.col("dup_key").is_duplicated().alias("dup"))

    # Generate the CHEMKIN strings for each reaction
    if schema.has_columns(rxn_df, ReactionSorted):
        cols = (
            "obj",
            "dup",
            ReactionSorted.pes,
            ReactionSorted.subpes,
            ReactionSorted.channel,
        )
        rxn_strs = [
            (
                text_with_comments(
                    reac.chemkin_string(o, dup=d, eq_width=eq_width),
                    f"pes.subpes.channel  {p}.{s}.{c}",
                )
                if any(x is not None for x in (p, s, c))
                else reac.chemkin_string(o, dup=d, eq_width=eq_width)
            )
            for o, d, p, s, c in rxn_df.select(*cols).rows()
        ]
    else:
        rxn_strs = [
            reac.chemkin_string(o, dup=d, eq_width=eq_width)
            for o, d in rxn_df.select("obj", "dup").rows()
        ]

    return block(KeyWord.REACTIONS, rxn_strs, header=header, frame=frame)


def block(key, val, header: str | None = None, frame: bool = True) -> str:
    """Write a block to a string.

    :param key: The starting key for the block
    :param val: The block value(s)
    :param header: A header for the block
    :param frame: Whether to frame the block with its header and footer
    :return: The block
    """
    start = key if header is None else f"{key} {header}"
    val = val if isinstance(val, str) else "\n".join(val)
    if not frame:
        return val
    return "\n\n".join([start, val, KeyWord.END])


def text_with_comments(text: str, comments: str, sep: str = "!") -> str:
    """Write text with comments to a combined string.

    :param text: Text
    :param comments: Comments
    :return: Combined text and comments
    """
    text_lines = text.splitlines()
    comm_lines = comments.splitlines()
    text_width = max(map(len, text_lines)) + 2

    lines = [
        f"{t:<{text_width}} {sep} {c}" if c else t
        for t, c in itertools.zip_longest(text_lines, comm_lines, fillvalue="")
    ]
    return "\n".join(lines)
