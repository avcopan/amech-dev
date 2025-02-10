"""Functions for writing CHEMKIN-formatted files."""

import functools
import itertools
from pathlib import Path

import automol
import polars
from autochem import rate, unit_
from autochem.unit_ import UNITS
from autochem.util import chemkin

from ... import reac_table, schema
from ..._mech import Mechanism
from ...schema import ReactionSorted, Species, SpeciesThermo
from ...util import col_
from .read import KeyWord

ENERGY_PER_SUBSTANCE_UNIT = unit_.string(UNITS.energy_per_substance).upper()
SUBSTANCE_UNIT = unit_.string(UNITS.substance).upper()


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
    fmls = list(map(automol.amchi.formula, mech.species[Species.amchi].to_list()))
    elem_strs = set(itertools.chain(*(f.keys() for f in fmls)))
    elem_strs = automol.form.sorted_symbols(elem_strs)
    return block(KeyWord.ELEMENTS, elem_strs)


def species_block(mech: Mechanism) -> str:
    """Write the species block to a string.

    :param mech: A mechanism
    :return: The species block string
    """
    name_width = 1 + mech.species[Species.name].str.len_chars().max()
    smi_width = 1 + mech.species[Species.smiles].str.len_chars().max()
    spc_strs = [
        f"{n:<{name_width}} ! SMILES: {s:<{smi_width}} AMChI: {c}"
        for n, s, c in mech.species.select(
            Species.name, Species.smiles, Species.amchi
        ).rows()
    ]
    return block(KeyWord.SPECIES, spc_strs)


def thermo_block(mech: Mechanism) -> str:
    """Write the thermo block to a string.

    :param mech: A mechanism
    :return: The thermo block string
    """
    if SpeciesThermo.thermo_string not in mech.species:
        return None

    # Generate the thermo strings
    therm_strs = mech.species.select(
        polars.concat_str(
            polars.col(Species.name).str.pad_end(24),
            polars.col(SpeciesThermo.thermo_string),
        )
    ).to_series()

    # Generate the header
    thermo_temps = mech.thermo_temps
    if thermo_temps is None:
        header = None
    else:
        thermo_temps_str = "  ".join(f"{t:.3f}" for t in thermo_temps)
        header = f"ALL\n    {thermo_temps_str}"

    return block(KeyWord.THERM, therm_strs, header=header)


def reactions_block(mech: Mechanism, frame: bool = True) -> str:
    """Write the reactions block to a string.

    :param mech: A mechanism
    :param frame: Whether to frame the block with its header and footer
    :return: The reactions block string
    """
    # Generate the header
    header = f"   {ENERGY_PER_SUBSTANCE_UNIT}   {SUBSTANCE_UNIT}"

    rxn_df = mech.reactions

    # Quit if no reactions
    if rxn_df.is_empty():
        return block(KeyWord.REACTIONS, "", header=header, frame=frame)

    # Identify duplicates
    dup_col = col_.temp()
    rxn_df = reac_table.with_duplicate_column(rxn_df, dup_col)

    # Add reaction objects
    obj_col = col_.temp()
    rxn_df = reac_table.with_rate_objects(rxn_df, obj_col, fill=True)

    # Add reaction equations to determine apppropriate width
    eq_col = col_.temp()
    rxn_df = rxn_df.with_columns(
        polars.col(obj_col)
        .map_elements(rate.chemkin_equation, return_dtype=polars.String)
        .alias(eq_col)
    )
    eq_width = 8 + rxn_df.get_column(eq_col).str.len_chars().max()

    # Add Chemkin rate strings
    ck_col = col_.temp()
    chemkin_string_ = functools.partial(rate.chemkin_string, eq_width=eq_width)
    rxn_df = rxn_df.with_columns(
        polars.col(obj_col)
        .map_elements(chemkin_string_, return_dtype=polars.String)
        .alias(ck_col)
    )

    # Add duplicate keywords
    rxn_df = rxn_df.with_columns(
        polars.when(dup_col)
        .then(
            polars.col(ck_col).map_elements(
                chemkin.write_with_dup, return_dtype=polars.String
            )
        )
        .otherwise(polars.col(ck_col))
    )

    # Add sort parameters
    srt_col = col_.temp()
    srt_expr = (
        polars.concat_list(schema.columns(ReactionSorted))
        if schema.has_columns(rxn_df, ReactionSorted)
        else polars.lit([None, None, None])
    )
    rxn_df = rxn_df.with_columns(srt_expr.alias(srt_col))

    # Get strings
    rxn_strs = [
        (text_with_comments(r, f"pes.subpes.channel  {'.'.join(s)}") if any(s) else r)
        for r, s in rxn_df.select(ck_col, srt_col).rows()
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
