"""Functions acting on species DataFrames."""

from collections.abc import Mapping, Sequence
from typing import Annotated

import automol
import polars
import pydantic
from pydantic_core import core_schema

import autochem as ac

from .util import c_, df_, pandera_
from .util.pandera_ import Model


class Species(Model):
    """Core species table."""

    name: str
    smiles: str
    amchi: str
    spin: int
    charge: int
    formula: polars.Struct


class SpeciesTherm(Model):
    """Species table with thermo."""

    therm: polars.Struct


assert all(
    f in pandera_.columns([Species, SpeciesTherm])
    for f in ac.therm.Species.model_fields
), "Make sure field names match autochem."


class SpeciesThermoOld(Model):
    """Species table with thermo."""

    thermo_string: str


class SpeciesStereo(Model):
    """Stereo-expanded species table."""

    orig_name: str
    orig_smiles: str
    orig_amchi: str


KEY_COLS = (Species.amchi, Species.spin, Species.charge)


# validation
def validate(
    df: polars.DataFrame, model_: Model | Sequence[Model] = ()
) -> polars.DataFrame:
    """Validate species DataFrame against model(s).

    :param df: DataFrame
    :param model_: Model(s)
    :return: DataFrame
    """
    models = [Species, *pandera_.normalize_model_input(model_)]
    return pandera_.validate(models, df)


SpeciesDataFrame_ = Annotated[
    pydantic.SkipValidation[polars.DataFrame],
    pydantic.BeforeValidator(polars.DataFrame),
    pydantic.AfterValidator(validate),
    pydantic.PlainSerializer(lambda x: polars.DataFrame(x).to_dict(as_series=False)),
    pydantic.GetPydanticSchema(
        lambda _, handler: core_schema.with_default_schema(handler(dict[str, list]))
    ),
]


# properties
def amchis(
    spc_df: polars.DataFrame,
    vals_: Sequence[object | Sequence[object]] | None = None,
    col_: str | Sequence[str] = Species.name,
    fill: bool = False,
) -> list[str]:
    """Get IDs for a species DataFrame.

    :param spc_df: Species DataFrame
    :param vals_: Optionally, lookup IDs for species matching these column value(s)
    :param col_: Column name(s) corresponding to `vals_`
    :param try_fill: Whether to attempt to fill missing values
    :return: Species IDs
    """
    vals_lst, cols = normalize_values_arguments(vals_, col_)
    chis = df_.values(spc_df, Species.amchi, vals_in_=vals_lst, col_in_=cols)
    if fill and cols[0] == Species.amchi:
        chis = [v[0] for v in vals_lst]

    return chis


# update
def update(
    spc_df1: polars.DataFrame,
    spc_df2: polars.DataFrame,
    key_col_: str | Sequence[str] = KEY_COLS,
    drop_orig: bool = True,
    how: str = "full",
) -> polars.DataFrame:
    """Update species data by species key.

    :param spc_df1: Species DataFrame
    :param spc_df1: Species DataFrame to update from
    :param key_col_: Species key column(s)
    :param drop_orig: Whether to drop the original column values
    :param how: Polars join strategy
    :return: Species DataFrame
    """
    return df_.update(spc_df1, spc_df2, col_=key_col_, drop_orig=drop_orig, how=how)


def left_update(
    spc_df1: polars.DataFrame,
    spc_df2: polars.DataFrame,
    key_col_: str | Sequence[str] = KEY_COLS,
    drop_orig: bool = True,
) -> polars.DataFrame:
    """Left-update species data by species key.

    :param spc_df1: Species DataFrame
    :param spc_df1: Species DataFrame to update from
    :param key_col_: Species key column(s)
    :param drop_orig: Whether to drop the original column values
    :return: Species DataFrame
    """
    return update(spc_df1, spc_df2, key_col_=key_col_, drop_orig=drop_orig, how="left")


# add columns
def with_key(
    spc_df: polars.DataFrame, col: str = "key", stereo: bool = True
) -> polars.DataFrame:
    """Add a key for identifying unique species.

    The key is "{AMChI}_{spin}_{charge}"

    :param spc_df: Species DataFrame
    :param col: Column name, defaults to "key"
    :param stereo: Whether to include stereochemistry
    :return: Species DataFrame
    """
    id_cols = KEY_COLS

    tmp_col = c_.temp()
    if not stereo:
        spc_df = df_.map_(spc_df, Species.amchi, tmp_col, automol.amchi.without_stereo)
        id_cols = (tmp_col, *id_cols[1:])

    spc_df = df_.with_concat_string_column(spc_df, col_out=col, col_=id_cols)
    if not stereo:
        spc_df = spc_df.drop(tmp_col)
    return spc_df


def with_therm_objects(spc_df: polars.DataFrame, col: str) -> polars.DataFrame:
    """Get reaction rate objects as a list.

    :param spc_df: Species DataFrame
    :param col: Column
    :return: Rate objects
    """
    cols = [
        Species.name,
        SpeciesTherm.therm,
    ]
    return spc_df.with_columns(
        polars.struct(cols)
        # .map_elements(ac.therm.Species.model_validate, return_dtype=polars.Object)
        .map_elements(print, return_dtype=polars.Object).alias(col)
    )


# tranform
def rename(
    spc_df: polars.DataFrame,
    names: Sequence[str] | Mapping[str, str],
    new_names: Sequence[str] | None = None,
    drop_orig: bool = True,
) -> polars.DataFrame:
    """Rename species in a species DataFrame.

    :param rxn_df: Species DataFrame
    :param names: A list of names or mapping from current to new names
    :param new_names: A list of new names
    :param drop_orig: Whether to drop the original names, or include them as `orig`
    :return: Species DataFrame
    """
    col_dct = c_.to_orig(Species.name)
    spc_df = spc_df.with_columns(polars.col(c0).alias(c) for c0, c in col_dct.items())
    expr = polars.col(Species.name)
    expr = expr.replace(names) if new_names is None else expr.replace(names, new_names)
    spc_df = spc_df.with_columns(expr)
    if drop_orig:
        spc_df = spc_df.drop(col_dct.values())
    return spc_df


# sort
def sort_by_formula(spc_df: polars.DataFrame) -> polars.DataFrame:
    """Sort species by formula.

    :param spc_df: Species DataFrame
    :return: Species DataFrame, sorted by formula
    """
    all_atoms = [s for s, *_ in spc_df.schema[Species.formula]]
    heavy_atoms = [s for s in all_atoms if s != "H"]
    return spc_df.sort(
        polars.sum_horizontal(
            polars.col(Species.formula).struct.field(*heavy_atoms)
        ),  # heavy atoms
        polars.sum_horizontal(
            polars.col(Species.formula).struct.field(*all_atoms)
        ),  # all atoms
        polars.col(Species.formula),
        nulls_last=True,
    )


# select
def filter(  # noqa: A001
    spc_df: polars.DataFrame,
    vals_: Sequence[object | Sequence[object]] | None = None,
    col_: str | Sequence[str] = Species.name,
) -> polars.DataFrame:
    """Filter to include only rows that match one or more species.

    :param spc_df: A species DataFrame
    :param col_name: The column name
    :param vals_lst: Column values list
    :param keys: Column keys
    :return: The modified species DataFrame
    """
    match_exprs = [species_match_expression(val_, col_) for val_ in vals_]
    return spc_df.filter(polars.any_horizontal(*match_exprs))


# helpers
def species_match_expression(
    val_: object | Sequence[object],
    key_: str | Sequence[str] = Species.name,
) -> polars.Expr:
    """Prepare a dictionary of species match data.

    :param val_: Column values
    :param key_: Column keys
    """
    if isinstance(key_, str):
        key_ = [key_]
        val_ = [val_]

    match_data = dict(zip(key_, val_, strict=True))
    if Species.smiles in match_data:
        match_data[Species.amchi] = automol.smiles.amchi(match_data.pop(Species.smiles))

    return polars.all_horizontal(*(polars.col(k) == v for k, v in match_data.items()))


def normalize_values_arguments(
    vals_: Sequence[object | Sequence[object]] | None = None,
    col_: str | Sequence[str] = Species.name,
) -> tuple[list[object], list[str]]:
    """Normalize species values input.

    Converts SMILES to AMChI if present.

    :param vals_: Optionally, lookup IDs for species matching these column value(s)
    :param col_: Column name(s) corresponding to `vals_`
    :return: Normalized value(s) list and column(s)
    """
    vals_, col_ = df_.normalize_values_arguments(vals_=vals_, col_=col_)

    # If using SMILES, convert to AMChI
    if Species.smiles in col_:
        smi_idx = col_.index(Species.smiles)
        col_[smi_idx] = Species.amchi
        if vals_:
            col_vals_ = list(zip(*vals_, strict=True))
            col_vals_[smi_idx] = list(map(automol.smiles.amchi, col_vals_[smi_idx]))
            vals_ = list(zip(*col_vals_, strict=True))

    return vals_, col_


def expand_stereo(
    spc_df: polars.DataFrame, enant: bool = True, strained: bool = False
) -> polars.DataFrame:
    """Stereoexpand species from mechanism.

    :param spc_df: Species table, as DataFrame
    :param enant: Distinguish between enantiomers?
    :param strained: Include strained stereoisomers?
    :return: Stereoexpanded species table
    """

    # Do species expansion based on AMChIs
    def _expand_amchi(chi):
        """Expand stereo for AMChIs."""
        return automol.amchi.expand_stereo(chi, enant=enant, strained=strained)

    spc_df = spc_df.rename(c_.to_orig(Species.amchi))
    spc_df = df_.map_(
        spc_df, c_.orig(Species.amchi), Species.amchi, _expand_amchi, bar=True
    )
    spc_df = spc_df.explode(polars.col(Species.amchi))

    # Update species names
    def _stereo_name(orig_name, chi):
        """Determine stereo name from AMChI."""
        return automol.amchi.chemkin_name(chi, root_name=orig_name)

    spc_df = spc_df.rename(c_.to_orig(Species.name))
    spc_df = df_.map_(
        spc_df, (c_.orig(Species.name), Species.amchi), Species.name, _stereo_name
    )

    # Update SMILES strings
    def _stereo_smiles(chi):
        """Determine stereo smiles from AMChI."""
        return automol.amchi.smiles(chi)

    spc_df = spc_df.rename(c_.to_orig(Species.smiles))
    spc_df = df_.map_(spc_df, Species.amchi, Species.smiles, _stereo_smiles, bar=True)
    return spc_df


# Bootstrapping function
def bootstrap(
    data: dict[str, Sequence[object]] | polars.DataFrame,
    name_dct: dict[str, str] | None = None,
    key: str = Species.amchi,
) -> polars.DataFrame:
    """Bootstrap species DataFrame from minimal data.

    :param data: Data
    :param name_dct: Names by key
    :param key: Key for filling from dictionaries, 'smiles' or 'amchi'.
    :return: DataFrame
    """
    # Make dataframe from given data
    df = polars.DataFrame(data, strict=False)
    df = df.rename({c: str.lower(c) for c in df.columns})
    df = pandera_.impose_schema(Species, df)

    # If empty, return early
    if df.is_empty():
        df = pandera_.add_missing_columns(Species, df)
        return validate(df)

    # Get spin from multiplicity if given
    if Species.spin not in df and "mult" in df:
        df = df.with_columns((polars.col("mult") - 1).cast(int).alias(Species.spin))

    # Add missing spin column
    df = pandera_.add_missing_columns(Species, df, Species.spin)

    # Sanitize SMILES with spin tags
    if Species.smiles in df:
        spin_tags = {"singlet": 0, "triplet": 2}
        for spin_tag, spin in spin_tags.items():
            # Use spin tag to populate spin column
            has_tag = polars.col(Species.smiles).str.contains(spin_tag)
            needs_spin = polars.col(Species.spin).is_null()
            spin0 = polars.col(Species.spin)
            get_spin = polars.when(has_tag & needs_spin).then(spin).otherwise(spin0)
            df = df.with_columns(get_spin.alias(Species.spin))
            # Remove spin tag from smiles column
            df = df.with_columns(polars.col(Species.smiles).str.replace(spin_tag, ""))

    # Populate AMChIs from SMILES or InChI
    if Species.amchi not in df and Species.smiles in df:
        df = df_.map_(df, Species.smiles, Species.amchi, automol.smiles.amchi, bar=True)
    if Species.amchi not in df and "inchi" in df:
        df = df_.map_(df, "inchi", Species.amchi, automol.inchi.amchi, bar=True)

    # Add spin where missing
    guess_spin = polars.col(Species.amchi).map_elements(
        automol.amchi.guess_spin, return_dtype=int
    )
    orig_spin = polars.col(Species.spin)
    expr = polars.when(orig_spin.is_null()).then(guess_spin).otherwise(orig_spin)
    df = df.with_columns(expr.alias(Species.spin))

    # Populate missing columns from AMChI
    populators = {
        Species.name: automol.amchi.chemkin_name,
        Species.smiles: automol.amchi.smiles,
        Species.charge: (lambda _: 0),
        Species.formula: automol.amchi.formula,
    }
    for col, pop_ in populators.items():
        if col not in df:
            dtype = pandera_.dtype(Species, col)
            expr = polars.col(Species.amchi).map_elements(pop_, return_dtype=dtype)
            df = df.with_columns(expr.alias(col))

    # Replace names if given
    if name_dct is not None:
        assert key in (Species.smiles, Species.amchi), f"Invalid key: {key}"
        if key == Species.smiles:
            name_dct = {automol.smiles.amchi(k): v for k, v in name_dct.items()}
        orig = polars.col(Species.name)
        expr = polars.col(Species.amchi).replace_strict(name_dct, default=orig)
        df = df.with_columns(expr.alias(Species.name))

    return validate(df)
