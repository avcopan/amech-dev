"""Functions for writing Mechanalyzer-formatted files."""

import io
from pathlib import Path

import automol
import pandas
import polars

from ..._mech import Mechanism
from ...schema import Model, Species, table_with_columns_from_models
from ...util import df_
from ..chemkin import write as chemkin_write


class MASpecies(Model):
    """Mechanalyzer species table."""

    name: str
    smiles: str
    inchi: str
    inchikey: str
    mult: str
    charge: str
    canon_enant_ich: str


def mechanism(
    mech: Mechanism,
    rxn_out: str | Path | None = None,
    spc_out: str | Path | None = None,
    string: bool = True,  # Change default to False when implemented
) -> tuple[str | dict, str | pandas.DataFrame]:
    """Write mechanism to MechAnalyzer format.

    :param mech: A mechanism
    :param rxn_out: Optionally, write the output to this file path (reactions)
    :param spc_out: Optionally, write the output to this file path (species)
    :param string: Return as Mechanalyzer CHEMKIN and CSV strings, instead of a reaction
        dictionary and species dataframe?
    :return: MechaAnalyzer reaction dictionary (or CHEMKIN string) and species dataframe
    """
    mech_str = chemkin_write.reactions_block(mech)
    if rxn_out is not None:
        rxn_out: Path = Path(rxn_out)
        rxn_out.write_text(mech_str)

    spc_ret = species(mech, out=spc_out, string=string)
    if string:
        return mech_str, spc_ret
    else:
        raise NotImplementedError("Writing to rxn_params_dct not yet implemented!")


def species(
    mech: Mechanism,
    out: str | None = None,
    string: bool = True,  # Change default to False when implemented
) -> str | pandas.DataFrame:
    """Write the species in a mechanism to a MechAnalyzer species table.

    :param mech: A mechanism
    :param out: Optionally, write the output to this file path (species)
    :param string: Return as Mechanalyzer CSV string, instead of a species dataframe?
    :return: A MechAnalyzer species dataframe
    """
    # Write species
    spc_df = mech.species
    spc_df = df_.map_(
        spc_df, Species.amchi, MASpecies.inchi, automol.amchi.chi_, bar=True
    )
    spc_df = df_.map_(
        spc_df, MASpecies.inchi, MASpecies.inchikey, automol.chi.inchi_key, bar=True
    )
    spc_df = df_.map_(
        spc_df,
        MASpecies.inchi,
        MASpecies.canon_enant_ich,
        automol.amchi.canonical_enantiomer,
        bar=True,
    )
    spc_df = spc_df.with_columns((polars.col(Species.spin) + 1).alias(MASpecies.mult))
    spc_df = table_with_columns_from_models(
        spc_df, model_=[MASpecies], keep_extra=False
    )
    if out is not None:
        df_.to_csv(spc_df, out, quote_char="'")

    spc_ret = spc_df.to_pandas()
    if string:
        spc_ret_io = io.StringIO()
        spc_ret.to_csv(spc_ret_io, quotechar="'", index=False)
        spc_ret = spc_ret_io.getvalue()
        spc_ret_io.close()

    return spc_ret
