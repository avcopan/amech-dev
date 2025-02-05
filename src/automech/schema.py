"""DataFrame schema."""

import itertools
from collections.abc import Sequence

import automol
import more_itertools as mit
import pandera.polars as pa
import polars
from polars.datatypes import Struct
from pydantic import BaseModel, ConfigDict

import automech.util.col_
from automech.util import df_

Model = pa.DataFrameModel


# Species table schemas
class Species(Model):
    """Core species table."""

    name: str
    smiles: str
    amchi: str
    spin: int
    charge: int
    formula: Struct


class SpeciesThermo(Model):
    """Species table with thermo."""

    thermo_string: str


class SpeciesStereo(Model):
    """Stereo-expanded species table."""

    orig_name: str
    orig_smiles: str
    orig_amchi: str


SPECIES_MODELS = (Species, SpeciesThermo, SpeciesStereo)


# Reaction table schemas
class Reaction(Model):
    """Core reaction table."""

    reactants: list[str]
    products: list[str]
    formula: Struct


class ReactionRateOld(Model):
    """Reaction table with rate."""

    rate: Struct
    colliders: Struct


class ReactionSorted(Model):
    """Reaction table with sort information."""

    pes: int
    subpes: int
    channel: int


class ReactionStereo(Model):
    """Stereo-expanded reaction table."""

    amchi: str
    orig_reactants: str
    orig_products: str


class ReactionCheck(Model):
    """Consistency checks for reaction table."""

    is_missing_species: bool
    has_unbalanced_formula: bool


class SpeciesMisc(Model):
    """Miscellaneous species columns (not for validation)."""

    orig_thermo_string: str


REACTION_MODELS = (Reaction, ReactionRateOld, ReactionStereo, ReactionCheck)


# Error data structure
class Errors(BaseModel):
    """Error values."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    species: polars.DataFrame | None = None
    reactions: polars.DataFrame | None = None

    def is_empty(self) -> bool:
        """Determine whether the errors object is empty.

        :return: `True` if it is, `False` if it isn't
        """
        return (self.species is None or self.species.is_empty()) and (
            self.reactions is None or self.reactions.is_empty()
        )


def columns(model_: Model | Sequence[Model]) -> list[str]:
    """Get column names.

    :param model_: Model(s)
    :return: Schema, as a mapping of column names to types
    """
    model_ = model_ if isinstance(model_, Sequence) else [model_]
    return list(
        mit.unique_everseen(
            itertools.chain.from_iterable(m.to_schema().columns for m in model_)
        )
    )


def types(
    model_: Model | Sequence[Model], keys: Sequence[str] | None = None, py: bool = False
) -> dict[str, type]:
    """Get a dictionary mapping column names to type names.

    :param models: The models to get the schema for
    :param keys: Optionally, specify keys to include in the schema
    :param py: Whether to return Python types instead of Polars types.
    :return: The schema, as a mapping of column names to types
    """
    model_ = model_ if isinstance(model_, Sequence) else [model_]
    keys = None if keys is None else list(keys)

    type_dct = {}
    for model in model_:
        type_dct.update({k: v.dtype.type for k, v in model.to_schema().columns.items()})

    if keys is not None:
        type_dct = {k: v for k, v in type_dct.items() if k in keys}

    if py:
        type_dct = {k: v.to_python() for k, v in type_dct.items()}

    return type_dct


def species_types(keys: Sequence[str] | None = None) -> dict[str, type]:
    """Get a dictionary mapping column names to types for species tables.

    :param keys: Optionally, specify keys to include in the schema
    :return: The schema, as a mapping of column names to types
    """
    return types(SPECIES_MODELS, keys)


def reaction_types(keys: Sequence[str] | None = None) -> dict[str, type]:
    """Get a dictionary mapping column names to types for reaction tables.

    :param keys: Optionally, specify keys to include in the schema
    :return: The schema, as a mapping of column names onto types
    """
    return types(REACTION_MODELS, keys)


def species_table(
    df: polars.DataFrame,
    model_: Model | Sequence[Model] = (),
    name_dct: dict[str, str] | None = None,
    spin_dct: dict[str, int] | None = None,
    charge_dct: dict[str, int] | None = None,
    keep_extra: bool = True,
) -> polars.DataFrame:
    """Validate a species DataFrame.

    :param df: The DataFrame
    :param models: Extra species models to validate against
    :param name_dct: If generating names, specify some names by ChI
    :param spin_dct: If generating spins, specify some spins by ChI
    :param charge_dct: If generating charges, specify some charges by ChI
    :param keep_extra: Keep extra columns that aren't in the models?
    :return: The validated DataFrame
    """
    model_ = model_ if isinstance(model_, Sequence) else [model_]

    if Species not in model_:
        model_ = (Species, *model_)

    if df.is_empty():
        df = polars.DataFrame([], schema={**df.schema, **types(model_)})

    dt_dct = species_types()
    df = df.rename({k: str.lower(k) for k in df.columns})
    assert (
        Species.amchi in df or Species.smiles in df
    ), f"Must have either 'amchi' or 'smiles' column: {df}"

    # Sanitize SMILES, which may have things like 'singlet[CH2]'
    spin_smi_dct = {"singlet": 0, "triplet": 2}
    if Species.smiles in df:
        for spin_type in spin_smi_dct:
            df = df.with_columns(
                polars.col(Species.smiles).str.contains(spin_type).alias(spin_type)
            )
            df = df.with_columns(polars.col(Species.smiles).str.replace(spin_type, ""))

    if Species.amchi not in df:
        df = df_.map_(df, Species.smiles, Species.amchi, automol.smiles.amchi, bar=True)

    if Species.smiles not in df:
        df = df_.map_(df, Species.amchi, Species.smiles, automol.amchi.smiles, bar=True)

    if Species.formula not in df:
        dt = dt_dct[Species.formula]
        df = df_.map_(
            df, Species.amchi, Species.formula, automol.amchi.formula, dtype_=dt
        )

    if Species.name not in df:
        dt = dt_dct[Species.name]
        df = df_.map_(
            df,
            Species.amchi,
            Species.name,
            automol.amchi.chemkin_name,
            dtype_=dt,
            dct=name_dct,
        )

    if Species.spin not in df:
        dt = dt_dct[Species.spin]
        if "mult" in df:
            df = df.with_columns((df["mult"] - 1).alias(Species.spin).cast(dt))
        else:
            df = df_.map_(
                df,
                Species.amchi,
                Species.spin,
                automol.amchi.guess_spin,
                dtype_=dt,
                dct=spin_dct,
            )

    for spin_type, spin_val in spin_smi_dct.items():
        if spin_type in df:
            df = df.with_columns(
                spin=polars.when(polars.col(spin_type))
                .then(spin_val)
                .otherwise(polars.col(Species.spin))
            )
            df = df.drop(spin_type)

    if Species.charge not in df:
        dt = dt_dct[Species.charge]
        df = df_.map_(
            df, Species.amchi, Species.charge, lambda _: 0, dtype_=dt, dct=charge_dct
        )

    for model in model_:
        df = model.validate(df)

    return table_with_columns_from_models(df, model_=model_, keep_extra=keep_extra)


def reaction_table(
    df: polars.DataFrame,
    model_: Model | Sequence[Model] = (),
    spc_df: polars.DataFrame | None = None,
    keep_extra: bool = True,
    fail_on_error: bool = True,
) -> tuple[polars.DataFrame, Errors]:
    """Validate a reactions DataFrame.

    :param df: The DataFrame
    :param models: Extra reaction models to validate against
    :param spc_df: Optionally, pass in a species DataFrame for determining formulas
    :param keep_extra: Keep extra columns that aren't in the models?
    :param fail_on_error: Whether or not to raise an Exception if there is an error
    :return: The validated DataFrame, along with any errors
    """
    model_ = model_ if isinstance(model_, Sequence) else [model_]

    if Reaction not in model_:
        model_ = (Reaction, *model_)

    if df.is_empty():
        df = polars.DataFrame([], schema={**df.schema, **types(model_)})

    df = df.rename({k: str.lower(k) for k in df.columns})
    df = reaction_table_with_sorted_reagents(df)
    df = reaction_table_with_missing_species_check(df, spc_df=spc_df)
    df = reaction_table_with_formula(df, spc_df=spc_df, check=True)
    check_cols = [c for c in ReactionCheck.to_schema().columns if c in df]
    check_expr = polars.concat_list(check_cols).list.any()
    err_df = df.filter(check_expr)
    if fail_on_error:
        assert err_df.is_empty(), f"Encountered errors: {err_df}"

    err = Errors(reactions=err_df)
    df = df.filter(~check_expr)
    df = df.drop(check_cols)

    if ReactionRateOld.colliders in df:
        fill_val = polars.lit(
            {"M": None}, dtype=df_.dtype(df, ReactionRateOld.colliders)
        )
        df = df.with_columns(polars.col(ReactionRateOld.colliders).fill_null(fill_val))

    for model in model_:
        df = model.validate(df)

    df = table_with_columns_from_models(df, model_=model_, keep_extra=keep_extra)
    return df, err


def reaction_table_with_sorted_reagents(df: polars.DataFrame) -> polars.DataFrame:
    """Sort the reagents columns in the reaction table.

    :param df: A reactions table
    :return: The reactions table with sorted reagents
    """
    df = df.with_columns(polars.col(Reaction.reactants).list.sort())
    df = df.with_columns(polars.col(Reaction.products).list.sort())
    return df


def reaction_table_with_missing_species_check(
    df: polars.DataFrame, spc_df: polars.DataFrame | None = None
) -> polars.DataFrame:
    """Add a column to the reaction table that identifies missing species.

    (Only does anything if a species DataFrame is provided.)

    :param df: A reactions DataFrame
    :param spc_df: A species DataFrame
    :return: The reactions table with the missing species check column
    """
    if spc_df is None:
        return df.with_columns(
            polars.lit(False).alias(ReactionCheck.is_missing_species)
        )

    names = spc_df[Species.name]
    return df.with_columns(
        polars.concat_list(Reaction.reactants, Reaction.products)
        .list.eval(polars.element().is_in(names))
        .list.all()
        .not_()
        .alias(ReactionCheck.is_missing_species)
    )


def reaction_table_with_formula(
    df: polars.DataFrame, spc_df: polars.DataFrame | None = None, check: bool = False
) -> polars.DataFrame:
    """Determine reaction formulas from their reagents (reactants or products).

    (Only does anything if a species DataFrame is provided.)

    :param df: A reactions DataFrame
    :param spc_df: A species DataFrame
    :param check: Check that the reactant and product formulas are balanced
    :return: The reaction DataFrame with the new formula column
    """
    df = _reaction_table_with_formula(df, spc_df=spc_df)
    if check and spc_df is not None:
        col_tmp = automech.util.col_.temp()

        df = _reaction_table_with_formula(
            df, spc_df=spc_df, col_in=Reaction.products, col_out=col_tmp
        )
        df = df.with_columns(
            (polars.col(Reaction.formula) != polars.col(col_tmp)).alias(
                ReactionCheck.has_unbalanced_formula
            )
        )
        df = df.drop(col_tmp)

    return df


def _reaction_table_with_formula(
    df: polars.DataFrame,
    spc_df: polars.DataFrame | None = None,
    col_in: str = Reaction.reactants,
    col_out: str = Reaction.formula,
) -> polars.DataFrame:
    """Determine reaction formulas from their reagents (reactants or products).

    The species DataFrame must be provided for formulas to be determined. Otherwise,
    the formula column will be initialized with empty values.

    :param df: The DataFrame
    :param spc_df: Optionally, pass in a species DataFrame for determining formulas
    :param col_in: A column with lists of species names (reactants or products), used to
        determine the overall formula
    :param col_out: The name of the new formula column
    :return: The reaction DataFrame with the new formula column
    """
    # If the column already exists and we haven't passed in a species dataframe, make
    # sure we don't wipe it out
    if Reaction.formula in df and spc_df is None:
        return df

    dt_dct = reaction_types()
    dt = dt_dct.get(Reaction.formula)

    if spc_df is None:
        df = df.with_columns(polars.lit({"H": None}).alias(col_out))
    else:
        col_tmp = automech.util.col_.temp()
        spc_df = species_table(spc_df)
        names = spc_df[Species.name]
        formulas = spc_df[Species.formula]
        expr = polars.element().replace_strict(names, formulas, default={"H": None})
        df = df.with_columns(polars.col(col_in).list.eval(expr).alias(col_tmp))
        df = df_.map_(df, col_tmp, col_out, automol.form.join_sequence, dtype_=dt)
        df = df.drop(col_tmp)

    return df


def table_with_columns_from_models(
    df: polars.DataFrame, model_: Model | Sequence[Model] = (), keep_extra: bool = True
) -> polars.DataFrame:
    """Return a table with columns selected from models.

    :param df: The DataFrame
    :param model_: The model(s), defaults to ()
    :param keep_extra: Keep extra columns that aren't in the models?
    :return: The DataFrame selection
    """
    cols = columns(model_)
    if keep_extra:
        cols.extend(c for c in df.columns if c not in cols)
    return df.select(cols)


def has_columns(df: polars.DataFrame, model_: Model | Sequence[Model]) -> bool:
    """Determine whether a DataFrame has the columns in one or more models.

    :param df: The DataFrame
    :param model_: The model(s)
    :return: `True` if it does, `False` if it doesn't
    """
    return all(c in df for c in columns(model_))
