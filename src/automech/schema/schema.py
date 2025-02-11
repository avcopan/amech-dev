"""DataFrame schema."""

import automol
import pandera.polars as pa
import polars

from ..util.model_ import df_


class Species(pa.DataFrameModel):
    """Core species table."""

    name: str
    smiles: str
    amchi: str
    spin: int
    charge: int
    formula: polars.Struct


def species_initialize(df: polars.DataFrame) -> polars.DataFrame:
    """Initialize species data with missing columns for sanitization.

    :param df: DataFrame
    :return: DataFrame
    """
    if Species.spin not in df:
        # Get spin from multiplicity, if present, otherwise initialize it as null
        spin = polars.col("mult") - 1 if "mult" in df else polars.lit(None, dtype=int)
        df = df.with_columns(spin.alias(Species.spin))

    return df


def species_sanitize(df: polars.DataFrame) -> polars.DataFrame:
    """Sanitize species data.

    :param df: DataFrame
    :return: DataFrame
    """
    # Sanitize spin-annotated SMILES, e.g. 'singlet[CH2]'
    spin_tags = {"singlet": 0, "triplet": 2}

    if Species.smiles in df:
        for spin_tag, spin in spin_tags.items():
            # Use spin tag to populate spin column
            has_tag = polars.col(Species.smiles).str.contains(spin_tag)
            needs_spin = polars.col(Species.spin).is_null()
            spin0 = polars.col(Species.spin)
            get_spin = polars.when(has_tag & needs_spin).then(spin).otherwise(spin0)
            df = df.with_columns(get_spin.alias(Species.spin))
            # Remove spin tag from smiles column
            df = df.with_columns(polars.col(Species.smiles).str.replace(spin_tag, ""))
    return df


def species_fill(df: polars.DataFrame) -> polars.DataFrame:
    """Fill species data.

    :param df: DataFrame
    :return: DataFrame
    """
    # 1. Populate AMChIs from SMILES or InChI
    if Species.amchi not in df and Species.smiles in df:
        df = df_.map_(df, Species.smiles, Species.amchi, automol.smiles.amchi, bar=True)
    if Species.amchi not in df and "inchi" in df:
        df = df_.map_(df, "inchi", Species.amchi, automol.inchi.amchi, bar=True)

    assert Species.amchi in df, f"No species identifier detected: {df}"

    # 2. Fill in other data from AMChIs
    if Species.smiles not in df:
        df = df_.map_(df, Species.amchi, Species.smiles, automol.amchi.smiles, bar=True)
    if Species.formula not in df:
        df = df_.map_(df, Species.amchi, Species.formula, automol.amchi.formula)
