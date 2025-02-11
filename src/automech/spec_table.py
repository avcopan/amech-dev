"""Functions acting on species DataFrames."""

from collections.abc import Mapping, Sequence

import automol
import polars

from . import schema_old
from .schema_old import Species
from .util import col_, df_

m_col_ = col_

ID_COLS = (Species.amchi, Species.spin, Species.charge)
SpeciesId = tuple[*schema_old.types(Species, ID_COLS, py=True).values()]


# properties
def species_ids(
    spc_df: polars.DataFrame,
    vals_: Sequence[object | Sequence[object]] | None = None,
    col_: str | Sequence[str] = Species.name,
    try_fill: bool = False,
) -> list[SpeciesId]:
    """Get IDs for a species DataFrame.

    :param spc_df: Species DataFrame
    :param vals_: Optionally, lookup IDs for species matching these column value(s)
    :param col_: Column name(s) corresponding to `vals_`
    :param try_fill: Whether to attempt to fill missing values
    :return: Species IDs
    """
    spc_id_col_ = ID_COLS
    vals_, col_ = normalize_values_arguments(vals_, col_)
    vals_out_ = df_.values(spc_df, spc_id_col_, vals_in_=vals_, col_in_=col_)

    spc_ids = []
    for val_, val_out_ in zip(vals_, vals_out_, strict=True):
        spc_ids.append(
            species_id_fill_value(val_, col_)
            if try_fill and not any(val_out_)
            else val_out_
        )
    return list(map(tuple, spc_ids))


def species_names_by_id(
    spc_df: polars.DataFrame, spc_ids: Sequence[SpeciesId]
) -> list[str]:
    """Get species names by ID from a species DataFrame.

    :param spc_df: Species DataFrame
    :param spc_ids: Species IDs (AMChI, spin, charge)
    :return: Species names
    """
    spc_id_col_ = ID_COLS
    return df_.values(spc_df, Species.name, vals_in_=spc_ids, col_in_=spc_id_col_)


# update
def update(
    spc_df: polars.DataFrame,
    src_spc_df: polars.DataFrame,
    key_col_: str | Sequence[str] = ID_COLS,
) -> polars.DataFrame:
    """Left-update species data by species key.

    :param spc_df: Species DataFrame
    :param src_spc_df: Source species DataFrame
    :param key_col_: Species key column(s)
    :return: Species DataFrame
    """
    return df_.update(spc_df, src_spc_df, col_=key_col_)


def left_update(
    spc_df: polars.DataFrame,
    src_spc_df: polars.DataFrame,
    key_col_: str | Sequence[str] = ID_COLS,
    drop_orig: bool = True,
) -> polars.DataFrame:
    """Left-update species data by species key.

    :param spc_df: Species DataFrame
    :param src_spc_df: Source species DataFrame
    :param key_col_: Species key column(s)
    :param drop_orig: Whether to drop the original column values
    :return: Species DataFrame
    """
    return df_.left_update(spc_df, src_spc_df, col_=key_col_, drop_orig=drop_orig)


# add rows
def add_missing_species_by_id(
    spc_df: polars.DataFrame, spc_ids: Sequence[SpeciesId]
) -> polars.DataFrame:
    """Add missing species to a species DataFrame.

    :param spc_df: Species DataFrame
    :param spc_ids: Species IDs
    :return: Species DataFrame
    """
    id_col_ = ID_COLS
    idx_col = col_.temp()
    spc_df = df_.with_match_index_column(spc_df, idx_col, vals_=spc_ids, col_=id_col_)

    miss_spc_ids = [s for i, s in enumerate(spc_ids) if i not in spc_df[idx_col]]
    miss_spc_df = polars.DataFrame(miss_spc_ids, schema=id_col_, orient="row")
    miss_spc_df = schema_old.species_table(miss_spc_df)
    return polars.concat([spc_df.drop(idx_col), miss_spc_df], how="diagonal_relaxed")


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
    id_cols = ID_COLS

    tmp_col = col_.temp()
    if not stereo:
        spc_df = df_.map_(spc_df, Species.amchi, tmp_col, automol.amchi.without_stereo)
        id_cols = (tmp_col, *id_cols[1:])

    spc_df = df_.with_concat_string_column(spc_df, col_out=col, col_=id_cols)
    if not stereo:
        spc_df = spc_df.drop(tmp_col)
    return spc_df


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
    col_dct = col_.to_orig(Species.name)
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


def species_id_fill_value(
    val_: Sequence[object | Sequence[object]] | None = None,
    col_: str | Sequence[str] = Species.name,
) -> SpeciesId:
    """Calculate an appropriate fill value for a species ID.

    :param val_: Column value(s)
    :param col_: Column name(s)
    :return: Species ID
    """
    dct = dict(zip(col_, val_, strict=True))
    amchi = (
        automol.smiles.amchi(dct[Species.smiles])
        if Species.smiles in dct
        else dct.get(Species.amchi)
    )
    spin = dct[Species.spin] if Species.spin in dct else automol.amchi.guess_spin(amchi)
    charge = dct[Species.charge] if Species.charge in dct else 0
    return (amchi, spin, charge)


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

    spc_df = spc_df.rename(col_.to_orig(Species.amchi))
    spc_df = df_.map_(
        spc_df, col_.orig(Species.amchi), Species.amchi, _expand_amchi, bar=True
    )
    spc_df = spc_df.explode(polars.col(Species.amchi))

    # Update species names
    def _stereo_name(orig_name, chi):
        """Determine stereo name from AMChI."""
        return automol.amchi.chemkin_name(chi, root_name=orig_name)

    spc_df = spc_df.rename(col_.to_orig(Species.name))
    spc_df = df_.map_(
        spc_df, (col_.orig(Species.name), Species.amchi), Species.name, _stereo_name
    )

    # Update SMILES strings
    def _stereo_smiles(chi):
        """Determine stereo smiles from AMChI."""
        return automol.amchi.smiles(chi)

    spc_df = spc_df.rename(col_.to_orig(Species.smiles))
    spc_df = df_.map_(spc_df, Species.amchi, Species.smiles, _stereo_smiles, bar=True)
    return spc_df
