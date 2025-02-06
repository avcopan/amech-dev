"""Functions for reading CHEMKIN-formatted files."""

import itertools
import re
from collections.abc import Collection

import more_itertools as mit
import pint
import polars
import pyparsing as pp
from pyparsing import common as ppc

import autochem
from autochem.util import unit_

from ... import data, schema, spec_table
from ..._mech import Mechanism
from ..._mech import from_data as mechanism_from_data
from ...schema import (
    Errors,
    Reaction,
    ReactionRate,
    ReactionRateOld,
    Species,
    SpeciesThermo,
)
from ...util import df_, io_
from ...util.io_ import TextInput, TextOutput


class KeyWord:
    # Blocks
    ELEMENTS = "ELEMENTS"
    THERM = "THERM"
    SPECIES = "SPECIES"
    REACTIONS = "REACTIONS"
    END = "END"
    # Units
    # # Energy (E) units
    CAL_MOLE = "CAL/MOLE"
    KCAL_MOLE = "KCAL/MOLE"
    JOULES_MOLE = "JOULES/MOLE"
    KJOULES_MOLE = "KJOULES/MOLE"
    KELVINS = "KELVINS"
    # # Prefactor (A) units
    MOLES = "MOLES"
    MOLECULES = "MOLECULES"


# generic
COMMENT_REGEX = re.compile(r"!.*$", flags=re.M)
HASH_COMMENT_REGEX = re.compile(r"# .*$", flags=re.M)
COMMENT_START = pp.Suppress(pp.Literal("!"))
COMMENT_END = pp.Suppress(pp.LineEnd())
COMMENT = COMMENT_START + ... + COMMENT_END
COMMENTS = pp.ZeroOrMore(COMMENT)

# units
E_UNIT = pp.Opt(
    pp.CaselessKeyword(KeyWord.CAL_MOLE)
    ^ pp.CaselessKeyword(KeyWord.KCAL_MOLE)
    ^ pp.CaselessKeyword(KeyWord.JOULES_MOLE)
    ^ pp.CaselessKeyword(KeyWord.KJOULES_MOLE)
    ^ pp.CaselessKeyword(KeyWord.KELVINS)
)
A_UNIT = pp.Opt(
    pp.CaselessKeyword(KeyWord.MOLES) ^ pp.CaselessKeyword(KeyWord.MOLECULES)
)


# mechanism
def mechanism(
    inp: TextInput, out: TextOutput = None, spc_out: TextOutput = None
) -> tuple[Mechanism, Errors]:
    """Extract the mechanism from a CHEMKIN file.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :param out: Optionally, write the output to this file path (reactions)
    :param spc_out: Optionally, write the output to this file path (species)
    :return: The mechanism dataclass
    """
    spc_df = species(inp, out=spc_out)
    rxn_df, err = reactions(inp, out=out, spc_df=spc_df)
    thermo_temps = thermo_temperatures(inp)
    mech = mechanism_from_data(
        rxn_inp=rxn_df, spc_inp=spc_df, thermo_temps=thermo_temps
    )
    return mech, err


# reactions
def reactions(
    inp: TextInput,
    spc_df: polars.DataFrame | None = None,
    out: TextOutput = None,
) -> tuple[polars.DataFrame, Errors]:
    """Extract reaction information as a dataframe from a CHEMKIN file.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :param units: Convert the rates to these units, if needed
    :param spc_df: A species dataframe to be used for validation
    :param out: Optionally, write the output to this file path
    :return: The reactions dataframe, along with any errors that were encountered
    """
    units = reactions_units(inp)

    def _is_reaction_line(string: str) -> bool:
        return re.search(r"\d\s*$", string)

    # Do the parsing
    rxn_block_str = reactions_block(inp, comments=False)
    line_iter = itertools.dropwhile(
        lambda s: not _is_reaction_line(s), rxn_block_str.splitlines()
    )
    rxn_strs = list(map("\n".join, mit.split_before(line_iter, _is_reaction_line)))
    rxns = [autochem.rate.from_chemkin_string(r, units=units) for r in rxn_strs]

    data_dct = {
        Reaction.reactants: [r.reactants for r in rxns],
        Reaction.products: [r.products for r in rxns],
        ReactionRate.reversible: [r.reversible for r in rxns],
        ReactionRate.rate_constant: [r.rate_constant.model_dump() for r in rxns],
    }
    schema_dct = schema.types([Reaction, ReactionRate], keys=data_dct.keys())
    rxn_df = polars.DataFrame(
        data=data_dct, schema=schema_dct, infer_schema_length=None
    )

    rxn_df, err = schema.reaction_table(
        rxn_df, spc_df=spc_df, model_=[Reaction, ReactionRate], fail_on_error=False
    )

    df_.to_csv(rxn_df, out)

    return rxn_df, err


def reactions_block(
    inp: TextInput, comments: bool = True, strip: bool = True
) -> str | None:
    """Get the reactions block, starting with 'REACTIONS' and ending in 'END'.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :param comments: Include comments?
    :param strip: Strip spaces from the ends?
    :return: The block
    """
    return block(inp, KeyWord.REACTIONS, comments=comments, strip=strip)


def reactions_units(inp: TextInput) -> unit_.Units:
    """Get the units for reaction rate constants.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :param default: Return default values, if missing?
    :return: The units for E and A, respectively
    """
    rxn_block_str = reactions_block(inp, comments=False, strip=False)
    assert isinstance(rxn_block_str, str), f"inp = {inp}"

    line1 = rxn_block_str.splitlines()[0]
    line1_units = list(map(pint.Unit, map(str.lower, line1.split())))

    substance = next((u for u in line1_units if u.is_compatible_with("mol")), "mol")
    energy_per_substance = next(
        (u for u in line1_units if u.is_compatible_with("cal/mol")), "cal/mol"
    )
    energy = unit_.dimension_unit(energy_per_substance, "energy")

    assert substance == unit_.dimension_unit(
        energy_per_substance, "substance"
    ), f"Incompatible units: {substance} !~ {energy_per_substance}"

    units = unit_.CHEMKIN_UNITS
    units = units.update({"energy": energy, "substance": substance})
    return units


# species
def species(inp: TextInput, out: TextOutput = None) -> polars.DataFrame:
    """Get the list of species, along with their comments.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :param out: Optionally, write the output to this file path
    :return: A species dataframe
    """
    species_name = pp.Word(pp.printables)
    word = pp.Word(pp.printables, exclude_chars=":")
    value = pp.Group(word + pp.Suppress(":") + word)
    values = pp.ZeroOrMore(value)
    comment_values = COMMENT_START + values + COMMENT_END
    entry = species_name("name") + comment_values("values")
    parser = pp.Suppress(...) + pp.OneOrMore(pp.Group(entry))

    spc_block_str = species_block(inp, comments=True)

    data_lst = [
        {Species.name: r.get("name"), **dict(r.get("values").as_list())}
        for r in parser.parse_string(spc_block_str)
    ]
    spc_df = polars.DataFrame(data_lst)

    therm_df = thermo(inp, spc_df=spc_df)
    spc_df = spc_df if therm_df is None else therm_df

    spc_df = schema.species_table(spc_df)

    df_.to_csv(spc_df, out)

    return spc_df


def species_block(inp: TextInput, comments: bool = True) -> str:
    """Get the species block, starting with 'SPECIES' and ending in 'END'.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The block
    """
    return block(inp, KeyWord.SPECIES, comments=comments)


def species_names(inp: TextInput) -> list[str]:
    """Get the list of species.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The species
    """
    parser = pp.OneOrMore(pp.Word(pp.printables))
    spc_block_str = species_block(inp, comments=False)
    return parser.parse_string(spc_block_str).as_list()


# therm
def thermo(
    inp: TextInput, spc_df: polars.DataFrame | None = None, out: TextOutput = None
) -> polars.DataFrame:
    """Get thermodynamic data as a dataframe.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :param spc_df: Optionally, join this to a species dataframe
    :return: A thermo dataframe
    """
    therm_dct = thermo_entry_dict(inp)
    if therm_dct is None:
        return None

    data = {
        Species.name: list(therm_dct.keys()),
        SpeciesThermo.thermo_string: list(therm_dct.values()),
    }
    therm_df = polars.DataFrame(data)
    if spc_df is not None:
        therm_df = spec_table.left_update(spc_df, therm_df, key_col_=Species.name)

    df_.to_csv(therm_df, out)

    return therm_df


def thermo_block(inp: TextInput, comments: bool = True) -> str:
    """Get the therm block, starting with 'THERM' and ending in 'END'.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The block
    """
    return block(inp, KeyWord.THERM, comments=comments)


def thermo_temperatures(inp: TextInput) -> list[float] | None:
    """Get the therm block temperatures.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The temperatures
    """
    therm_block_str = thermo_block(inp, comments=False)
    if therm_block_str is None:
        return None

    parser = therm_temperature_expression()
    temps = parser.parse_string(therm_block_str).as_list()
    return list(map(float, temps)) if temps else None


def thermo_entries(inp: TextInput) -> list[str]:
    """Get the therm block entries.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The entries
    """
    therm_block_str = thermo_block(inp, comments=False)
    if therm_block_str is None:
        return None

    parser = pp.Suppress(therm_temperature_expression()) + pp.OneOrMore(
        therm_entry_expression()
    )
    entries = parser.parse_string(therm_block_str).as_list()
    return entries


def thermo_entry_dict(inp: TextInput) -> dict[str, str]:
    """Get the therm block entries as a dictionary by species name.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: A dictionary mapping species names to therm block entries
    """
    entries = thermo_entries(inp)
    if entries is None:
        return None

    return dict(e.split(maxsplit=1) for e in entries)


# generic
def block(
    inp: TextInput, key: str, comments: bool = False, strip: bool = True
) -> str | None:
    """Get a keyword block, starting with a key and ending in 'END'.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :param key: The key that the block starts with
    :param comments: Include comments?
    :param strip: Strip spaces from the ends?
    :return: The block
    """
    inp = io_.read_text(inp)

    pattern = rf"{key[:4]}\S*(.*?){KeyWord.END}"
    match = re.search(pattern, inp, re.M | re.I | re.DOTALL)
    if not match:
        return None

    block_str = match.group(1)
    # Remove comments, if requested
    if not comments:
        block_str = without_comments(block_str)

    return block_str.strip() if strip else block_str


def without_comments(inp: TextInput) -> str:
    """Get a CHEMKIN string or substring with comments removed.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The string, without comments
    """
    inp = io_.read_text(inp)

    inp = re.sub(COMMENT_REGEX, "", inp)
    return re.sub(HASH_COMMENT_REGEX, "", inp)


def all_comments(inp: TextInput) -> list[str]:
    """Get all comments from a CHEMKIN string or substring.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The comments
    """
    inp = io_.read_text(inp)

    return re.findall(COMMENT_REGEX, inp)


def therm_temperature_expression() -> pp.ParseExpression:
    """Generate a pyparsing expression for the therm block temperatures."""
    return pp.Suppress(... + pp.Opt(pp.CaselessKeyword("ALL"))) + pp.Opt(ppc.number * 3)


def therm_entry_expression() -> pp.ParseExpression:
    """Generate a pyparsing expression for a therm entry."""
    return pp.Combine(
        therm_line_expression(1)
        + therm_line_expression(2)
        + therm_line_expression(3)
        + therm_line_expression(4)
    )


def therm_line_expression(num: int) -> pp.ParseExpression:
    """Generate a pyparsing expression for a therm line."""
    num = pp.Literal(f"{num}")
    end = pp.LineEnd()
    return pp.AtLineStart(pp.Combine(pp.SkipTo(num + end, include=True)))
