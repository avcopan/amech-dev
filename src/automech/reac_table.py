"""Functions acting on reactions DataFrames."""

import itertools
from collections.abc import Mapping, Sequence

import automol
import more_itertools as mit
import polars

from . import data, schema, spec_table
from .schema import Reaction, ReactionRate, Species
from .util import col_, df_

m_col_ = col_

DEFAULT_REAGENT_SEPARATOR = " + "

ID_COLS = (Reaction.reactants, Reaction.products)
ReactionId = tuple[*schema.types(Reaction, ID_COLS, py=True).values()]


# properties
def reaction_ids(rxn_df: polars.DataFrame, cross_sort: bool = True) -> list[ReactionId]:
    """Get IDs for a reactions DataFrame.

    :param rxn_df: Species DataFrame
    :param cross_sort: Whether to make keys direction-agnostic by cross-sorting reagents
    :return: Reaction IDs
    """
    rxn_id_col_ = ID_COLS
    rxn_df = df_.with_sorted_columns(rxn_df, col_=rxn_id_col_, cross_sort=cross_sort)
    return rxn_df.select(rxn_id_col_).rows()


def has_colliders(rxn_df: polars.DataFrame) -> bool:
    """Determine whether a reactions DataFrame has colliders.

    :param rxn_df: Reactions DataFrame
    :return: `True` if it does, `False` if not
    """
    return ReactionRate.colliders in rxn_df and df_.has_values(
        rxn_df.get_column(ReactionRate.colliders).struct.unnest()
    )


def has_rates(rxn_df: polars.DataFrame) -> bool:
    """Determine whether a reactions DataFrame has rates.

    :param rxn_df: Reactions DataFrame
    :return: `True` if it does, `False` if not
    """
    return ReactionRate.rate in rxn_df and df_.has_values(
        rxn_df.get_column(ReactionRate.rate).struct.unnest()
    )


def reagents(rxn_df: polars.DataFrame) -> list[list[str]]:
    """Get reagents as lists.

    :param rxn_df: A reactions DataFrame
    :return: The reagents
    """
    rcts = rxn_df.get_column(Reaction.reactants).to_list()
    prds = rxn_df.get_column(Reaction.products).to_list()
    return sorted(mit.unique_everseen(rcts + prds))


def species(rxn_df: polars.DataFrame) -> list[str]:
    """Get species in reactions DataFrame.

    :param rxn_df: Reactions DataFrame
    :return: Species names
    """
    rcts = rxn_df.get_column(Reaction.reactants).to_list()
    prds = rxn_df.get_column(Reaction.products).to_list()
    return sorted(mit.unique_everseen(itertools.chain.from_iterable(rcts + prds)))


def reagent_strings(
    rxn_df: polars.DataFrame, sep: str = DEFAULT_REAGENT_SEPARATOR
) -> list[str]:
    """Get reagents as strings.

    :param rxn_df: A reactions DataFrame
    :param sep: The separator for joining reagent strings
    :return: The reagents as strings
    """
    return [sep.join(r) for r in reagents(rxn_df)]


# update
def left_update(
    rxn_df: polars.DataFrame,
    src_rxn_df: polars.DataFrame,
    drop_orig: bool = True,
) -> polars.DataFrame:
    """Left-update reaction data by reaction key.

    :param rxn_df: reaction DataFrame
    :param src_rxn_df: Source reaction DataFrame
    :param drop_orig: Whether to drop original column values
    :return: Reaction DataFrame
    """
    drop_cols = m_col_.orig(schema.columns(Reaction))

    # Add reaction keys
    tmp_col = col_.temp()
    rxn_df = with_key(rxn_df, tmp_col)
    src_rxn_df = with_key(src_rxn_df, tmp_col)

    # Update
    rxn_df = df_.left_update(rxn_df, src_rxn_df, col_=tmp_col, drop_orig=drop_orig)

    # Drop unnecessary columns
    rxn_df = rxn_df.drop(tmp_col, *drop_cols, strict=False)
    return rxn_df


def left_update_rates(
    rxn_df: polars.DataFrame, src_rxn_df: polars.DataFrame
) -> polars.DataFrame:
    """Read thermochemical data from one dataframe into another.

    (AVC note: I think this can be deprecated and replaced with the more general
    function above...)

    :param rxn_df: Reactions DataFrame
    :param src_rxn_df: Reactions DataFrame with thermochemical data
    :return: reactions DataFrame
    """
    rxn_df = rxn_df.rename(col_.to_orig(ReactionRate.rate), strict=False)

    if has_colliders(rxn_df):
        raise NotImplementedError(
            f"Updating rates with colliders not yet implemented.\n{rxn_df}"
        )

    col_key = col_.temp()
    rxn_df = with_key(rxn_df, col=col_key)
    src_rxn_df = with_key(src_rxn_df, col=col_key)
    rxn_df = rxn_df.join(src_rxn_df, how="left", on=col_key)
    rxn_df = rxn_df.drop(col_key, polars.selectors.ends_with("_right"))
    rxn_df, *_ = schema.reaction_table(rxn_df, model_=ReactionRate)
    return rxn_df


# add/remove rows
def add_missing_reactions_by_id(
    rxn_df: polars.DataFrame, rxn_ids: Sequence[ReactionId]
) -> polars.DataFrame:
    """Add missing reactions to a reactions DataFrame.

    :param spc_df: Reactions DataFrame
    :param rxn_ids: Reaction IDs
    :return: Reactions DataFrame
    """
    # Sort reaction IDs
    rxn_ids = normalize_reaction_ids(rxn_ids)

    # Sort reagents
    id_cols0 = ID_COLS
    id_cols = col_.prefix(id_cols0, col_.temp())
    rxn_df = with_sorted_reagents(
        rxn_df, col_=id_cols0, col_out_=id_cols, cross_sort=True
    )

    # Add match index column
    idx_col = col_.temp()
    rxn_df = df_.with_match_index_column(rxn_df, idx_col, vals_=rxn_ids, col_=id_cols)
    rxn_df = rxn_df.drop(id_cols)

    # Append missing reactions to reactions DataFrame
    miss_rxn_ids = [s for i, s in enumerate(rxn_ids) if i not in rxn_df[idx_col]]
    miss_rxn_df = polars.DataFrame(miss_rxn_ids, schema=id_cols0, orient="row")
    miss_rxn_df, *_ = schema.reaction_table(miss_rxn_df)
    return polars.concat([rxn_df.drop(idx_col), miss_rxn_df], how="diagonal_relaxed")


def drop_self_reactions(rxn_df: polars.DataFrame) -> polars.DataFrame:
    """Drop self-reactions from reactions DataFrame.

    :param rxn_df: Reactions DataFrame
    :return: Reactions DataFrame
    """
    rcol0 = Reaction.reactants
    pcol0 = Reaction.products
    rcol, pcol = col_.prefix((rcol0, pcol0), col_.temp())
    rxn_df = with_sorted_reagents(
        rxn_df, col_=(rcol0, pcol0), col_out_=(rcol, pcol), cross_sort=False
    )
    rxn_df = rxn_df.filter(polars.col(rcol) != polars.col(pcol))
    return rxn_df.drop(rcol, pcol)


# add/remove columns
def with_key(
    rxn_df: polars.DataFrame,
    col: str = "key",
    spc_df: polars.DataFrame | None = None,
    cross_sort: bool = True,
    stereo: bool = True,
) -> polars.DataFrame:
    """Add a key for identifying unique reactions to this DataFrame.

    The key is formed by sorting within and across reactants and products to form a
    reaction ID and then joining to form a concatenated string.

    If a species DataFrame is passed in, the reaction keys will be name-agnostic.

    :param rxn_df: Reactions DataFrame
    :param col: Column name
    :param spc_df: Optional species DataFrame, for using unique species IDs
    :param cross_sort: Whether to sort the reaction direction
    :param stereo: Whether to include stereochemistry
    :return: A reactions DataFrame with this key as a new column
    """
    rct_col0 = Reaction.reactants
    prd_col0 = Reaction.products
    rct_col = col_.temp()
    prd_col = col_.temp()

    # If requested, use species keys instead of names
    if spc_df is not None:
        id_col = col_.temp()
        spc_df = spec_table.with_key(spc_df, id_col, stereo=stereo)
        rxn_df = translate_reagents(
            rxn_df,
            spc_df[Species.name],
            spc_df[id_col],
            rct_col=rct_col,
            prd_col=prd_col,
        )
        rct_col0 = rct_col
        prd_col0 = prd_col

    # Sort reagents
    rxn_df = with_sorted_reagents(
        rxn_df,
        col_=[rct_col0, prd_col0],
        col_out_=[rct_col, prd_col],
        cross_sort=cross_sort,
    )

    # Concatenate
    rxn_df = df_.with_concat_string_column(
        rxn_df, col, col_=[rct_col, prd_col], col_sep="=", list_sep="+"
    )
    return rxn_df.drop(rct_col, prd_col)


def with_sorted_reagents(
    rxn_df: polars,
    col_: Sequence[str] = (Reaction.reactants, Reaction.products),
    col_out_: Sequence[str] | None = None,
    cross_sort: bool = True,
) -> polars.DataFrame:
    """Generate sorted reagents columns.

    :param rxn_df: Reactions DataFrame
    :param col_: Reactant and product column(s)
    :param col_out_: Output reactant and product column(s), if different from input
    :param cross_sort: Whether to sort the reaction direction
    :return: Reactions DataFrame
    """
    col_out_ = col_ if col_out_ is None else col_out_
    assert len(col_) == 2, f"len({col_}) != 2"
    assert len(col_out_) == 2, f"len({col_out_}) != 2"
    return df_.with_sorted_columns(
        rxn_df, col_=col_, col_out_=col_out_, cross_sort=cross_sort
    )


def with_rates(rxn_df: polars.DataFrame) -> polars.DataFrame:
    """Add placeholder rate data to this DataFrame, if missing.

    This is mainly needed for ChemKin mechanism writing.

    :param rxn_df: Reaction DataFrame
    :return: Reaction DataFrame
    """
    rate0 = dict(data.rate.SimpleRate())
    coll0 = {"M": None}

    if ReactionRate.rate not in rxn_df:
        rxn_df = rxn_df.with_columns(polars.lit(rate0).alias(ReactionRate.rate))

    if ReactionRate.colliders not in rxn_df:
        rxn_df = rxn_df.with_columns(polars.lit(coll0).alias(ReactionRate.colliders))

    rxn_df = rxn_df.with_columns(polars.col(ReactionRate.rate).fill_null(rate0))
    rxn_df = rxn_df.with_columns(polars.col(ReactionRate.colliders).fill_null(coll0))
    return rxn_df


def without_rates(rxn_df: polars.DataFrame) -> polars.DataFrame:
    """Remove rate data from this DataFrame, if present.

    This is mainly needed for ChemKin mechanism writing.

    :param rxn_df: Reaction DataFrame
    :return: Reaction DataFrame
    """
    if ReactionRate.rate not in rxn_df:
        rxn_df = rxn_df.drop(ReactionRate.rate)

    if ReactionRate.colliders not in rxn_df:
        rxn_df = rxn_df.drop(ReactionRate.colliders)

    return rxn_df


def with_species_presence_column(
    rxn_df: polars.DataFrame, col: str, species_names: Sequence[str]
) -> polars.DataFrame:
    """Add a column indicating the presence of one or more species.

    :param rxn_df: A reactions DataFrame
    :param species_names: Species names
    :param col: The column name
    :return: The modified reactions DataFrame
    """
    return rxn_df.with_columns(
        polars.concat_list(Reaction.reactants, Reaction.products)
        .list.eval(polars.element().is_in(species_names))
        .list.any()
        .alias(col)
    )


def with_reagent_strings_column(
    rxn_df: polars.DataFrame, col: str, sep: str = DEFAULT_REAGENT_SEPARATOR
) -> polars.DataFrame:
    """Add a column containing the reagent strings on either side of the reaction.

    e.g. ["C2H6 + OH", "C2H5 + H2O"]

    :param rxn_df: A reactions DataFrame
    :param col: The column name
    :param sep: The separator for joining reagent strings
    :return: The reactions DataFrame with this extra column
    """
    return rxn_df.with_columns(
        polars.concat_list(
            polars.col(Reaction.reactants).list.join(sep),
            polars.col(Reaction.products).list.join(sep),
        ).alias(col)
    )


def rename(
    rxn_df: polars.DataFrame,
    names: Sequence[str] | Mapping[str, str],
    new_names: Sequence[str] | None = None,
    drop_orig: bool = True,
) -> polars.DataFrame:
    """Rename species in a reactions DataFrame.

    :param rxn_df: Reactions DataFrame
    :param names: A list of names or mapping from current to new names
    :param new_names: A list of new names
    :param drop_orig: Whether to drop the original names, or include them as `orig`
    :return: Reactions DataFrame
    """
    col_dct = col_.to_orig([Reaction.reactants, Reaction.products])
    rxn_df = rxn_df.with_columns(polars.col(c0).alias(c) for c0, c in col_dct.items())
    rxn_df = translate_reagents(rxn_df=rxn_df, trans=names, trans_into=new_names)
    if drop_orig:
        rxn_df = rxn_df.drop(col_dct.values())
    return rxn_df


def translate_reagents(
    rxn_df: polars.DataFrame,
    trans: Sequence[object] | Mapping[object, object],
    trans_into: Sequence[object] | None = None,
    rct_col: str = Reaction.reactants,
    prd_col: str = Reaction.products,
) -> polars.DataFrame:
    """Translate the reagent names in a reactions DataFrame.

    :param rxn_df: A reactions DataFrame
    :param trans: A translation mapping or a sequence of values to replace
    :param trans_into: If `trans` is a sequence, a sequence of values to replace by,
        defaults to None
    :param rct_col: The column name to use for the reactants
    :param prd_col: The column name to use for the products
    :return: The updated reactions DataFrame
    """
    expr = (
        polars.element().replace(trans)
        if trans_into is None
        else polars.element().replace(trans, trans_into)
    )

    return rxn_df.with_columns(
        polars.col(Reaction.reactants).list.eval(expr).alias(rct_col),
        polars.col(Reaction.products).list.eval(expr).alias(prd_col),
    )


def select_pes(
    rxn_df: polars.DataFrame,
    formula_: str | dict | Sequence[str | dict],
    exclude: bool = False,
) -> polars.DataFrame:
    """Select (or exclude) PES by formula(s).

    :param rxn_df: Reaction DataFrame
    :param formula_: PES formula(s) to include or exclude
    :param exclude: Whether to exclude or include the formula(s)
    :return: Reaction DataFrame
    """
    formula_ = [formula_] if isinstance(formula_, str | dict) else formula_
    fmls = [automol.form.from_string(f) if isinstance(f, str) else f for f in formula_]

    def _match(fml: dict[str, int]) -> bool:
        return any(automol.form.match(fml, f) for f in fmls)

    col_tmp = col_.temp()
    rxn_df = df_.map_(rxn_df, Reaction.formula, col_tmp, _match)
    match_expr = polars.col(col_tmp)
    rxn_df = rxn_df.filter(~match_expr if exclude else match_expr)
    rxn_df = rxn_df.drop(col_tmp)
    return rxn_df


# helpers
def normalize_reaction_ids(
    rxn_ids: Sequence[ReactionId], cross_sort: bool = True
) -> list[ReactionId]:
    """Normalize a list of reaction IDs.

    :param rxn_ids: Reaction IDs
    :param cross_sort: Whether to make keys direction-agnostic by cross-sorting reagents
    :return: Reaction IDs
    """
    return reaction_ids(
        polars.DataFrame(rxn_ids, schema=ID_COLS, orient="row"), cross_sort=cross_sort
    )
