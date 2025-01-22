"""Functions for reading Mechanalyzer-formatted files."""

import io
import os
from pathlib import Path

import pandas
import polars

from ..._mech import Mechanism
from ..._mech import from_data as mechanism_from_data
from ...schema import Errors, species_table
from ...util import df_
from ..chemkin import read as chemkin_read


def mechanism(
    rxn_inp: str,
    spc_inp: pandas.DataFrame | str | Path,
    rxn_out: str | None = None,
    spc_out: str | None = None,
) -> tuple[Mechanism, Errors]:
    """Extract the mechanism from MechAnalyzer files.

    :param rxn_inp: A mechanism (CHEMKIN format), as a file path or string
    :param spc_inp: A Mechanalyzer species table, as a file path or string or dataframe
    :param out: Optionally, write the output to this file path (reactions)
    :param spc_out: Optionally, write the output to this file path (species)
    :return: The mechanism dataclass
    """
    spc_df = species(spc_inp, out=spc_out)
    rxn_df, err = chemkin_read.reactions(rxn_inp, out=rxn_out, spc_df=spc_df)
    mech = mechanism_from_data(rxn_inp=rxn_df, spc_inp=spc_df)
    return mech, err


def species(
    inp: pandas.DataFrame | str | Path, out: str | None = None
) -> polars.DataFrame:
    """Extract species information as a dataframe from a Mechanalyzer species CSV.

    :param inp: A Mechanalyzer species CSV, as a file path or string
    :param out: Optionally, write the output to this file path
    :return: The species dataframe
    """
    if isinstance(inp, str | Path):
        inp = Path(inp).read_text() if os.path.exists(inp) else str(inp)
        spc_df = polars.read_csv(io.StringIO(inp), quote_char="'")
    else:
        assert isinstance(inp, pandas.DataFrame), f"Invalid species input: {inp}"
        spc_df = polars.from_pandas(inp)

    spc_df = species_table(spc_df)
    df_.to_csv(spc_df, out)

    return spc_df
