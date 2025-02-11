"""Definition and core functionality of mechanism data structure."""

import functools
import itertools
from collections.abc import Callable, Collection, Mapping, Sequence

import automol
import more_itertools as mit
import polars
import pydantic

import autochem

from . import data, reac_table, schema, schema_old, spec_table
from . import net as net_
from .schema.species import SpeciesDataFrame_
from .schema_old import (
    Reaction,
    ReactionRate,
    ReactionSorted,
    ReactionStereo,
    Species,
    SpeciesStereo,
)
from .util import col_, df_
from .util.type_ import DataFrame_


class Mechanism(pydantic.BaseModel):
    """Chemical kinetic mechanism."""

    reactions: DataFrame_
    species: SpeciesDataFrame_
    thermo_temps: tuple[float, float, float] | None = None


def from_network(net: net_.Network) -> Mechanism:
    """Generate mechanism from reaction network.

    :param net: Reaction network
    :return: Mechanism
    """
    spc_data = list(
        itertools.chain(*(d[net_.Key.species] for *_, d in net.nodes.data()))
    )
    rxn_data = [d for *_, d in net.edges.data()]

    spc_df = (
        polars.DataFrame([])
        if not spc_data
        else (
            polars.DataFrame(spc_data)
            .sort(net_.Key.id)
            .unique(net_.Key.id, maintain_order=True)
        )
    )
    rxn_df = (
        polars.DataFrame([])
        if not rxn_data
        else (
            polars.DataFrame(rxn_data)
            .sort(net_.Key.id)
            .unique(net_.Key.id, maintain_order=True)
        )
    )
    spc_df = spc_df.drop(net_.Key.id, strict=False)
    rxn_df = rxn_df.drop(net_.Key.id, strict=False)
    return Mechanism(reactions=rxn_df, species=spc_df)


def from_smiles(
    spc_smis: Sequence[str] = (),
    rxn_smis: Sequence[str] = (),
    name_dct: dict[str, str] | None = None,
    src_mech: Mechanism | None = None,
) -> Mechanism:
    """Generate mechanism using SMILES strings for species names.

    If `name_dct` is `None`, CHEMKIN names will be auto-generated.

    :param spc_smis: Species SMILES strings
    :param rxn_smis: Optionally, reaction SMILES strings
    :param name_dct: Optionally, specify name for some molecules
    :param spin_dct: Optionally, specify spin state (2S) for some molecules
    :param charge_dct: Optionally, specify charge for some molecules
    :param src_mech: Optional source mechanism for species names
    :return: Mechanism
    """
    # Add in any missing species from reaction SMILES
    spc_smis_by_rxn = [
        rs + ps
        for (rs, ps) in map(automol.smiles.reaction_reactants_and_products, rxn_smis)
    ]
    spc_smis = list(mit.unique_everseen(itertools.chain(spc_smis, *spc_smis_by_rxn)))

    # Build species dataframe
    spc_df = schema.species.bootstrap(
        {Species.smiles: spc_smis}, name_dct=name_dct, key=Species.smiles
    )

    # Left-update by species key, if source mechanism was provided
    if src_mech is not None:
        spc_df = spec_table.left_update(spc_df, src_mech.species, drop_orig=True)

    # Build reactions dataframe
    trans_dct = df_.lookup_dict(spc_df, Species.smiles, Species.name)
    rxn_smis_lst = list(map(automol.smiles.reaction_reactants_and_products, rxn_smis))
    data_lst = [
        {
            Reaction.reactants: list(map(trans_dct.get, rs)),
            Reaction.products: list(map(trans_dct.get, ps)),
        }
        for rs, ps in rxn_smis_lst
    ]
    dt = schema_old.reaction_types([Reaction.reactants, Reaction.products])
    rxn_df = polars.DataFrame(data=data_lst, schema=dt)

    mech = Mechanism(reactions=rxn_df, species=spc_df)
    return mech if src_mech is None else left_update(mech, src_mech)


# properties
def species_count(mech: Mechanism) -> int:
    """Get number of species in mechanism.

    :param mech: Mechanism
    :return: Number of species
    """
    return df_.count(mech.species)


def reaction_count(mech: Mechanism) -> int:
    """Get number of reactions in mechanism.

    :param mech: Mechanism
    :return: Number of reactions
    """
    return df_.count(mech.reactions)


def reagents(mech: Mechanism) -> list[list[str]]:
    """Get sets of reagents in mechanism.

    :param mech: Mechanism
    :return: Sets of reagents
    """
    return reac_table.reagents(mech.reactions)


def species_names(
    mech: Mechanism,
    rxn_only: bool = False,
    formulas: Sequence[str] | None = None,
    exclude_formulas: Sequence[str] = (),
) -> list[str]:
    """Get names of species in mechanism.

    :param mech: Mechanism
    :param rxn_only: Only include species that are involved in reactions?
    :param formulas: Formula strings of species to include, using * for wildcard
        stoichiometry
    :param exclude_formulas: Formula strings of species to exclude, using * for wildcard
        stoichiometry
    :return: Species names
    """

    def _formula_matcher(fml_strs):
        """Determine whether a species is excluded."""
        fmls = list(map(automol.form.from_string, fml_strs))

        def _matches_formula(chi):
            fml = automol.amchi.formula(chi)
            return any(automol.form.match(fml, e) for e in fmls)

        return _matches_formula

    spc_df = mech.species

    if formulas is not None:
        spc_df = df_.map_(
            spc_df, Species.amchi, "match", _formula_matcher(formulas), dtype_=bool
        )
        spc_df = spc_df.filter(polars.col("match"))

    if exclude_formulas:
        spc_df = df_.map_(
            spc_df, Species.amchi, "match", _formula_matcher(exclude_formulas)
        )
        spc_df = spc_df.filter(~polars.col("match"))

    spc_names = spc_df[Species.name].to_list()

    if rxn_only:
        rxn_df = mech.reactions
        rxn_spc_names = reac_table.species(rxn_df)
        spc_names = [n for n in spc_names if n in rxn_spc_names]

    return spc_names


def rename_dict(mech1: Mechanism, mech2: Mechanism) -> tuple[dict[str, str], list[str]]:
    """Generate dictionary for renaming species names from one mechanism to another.

    :param mech1: Mechanism with original names
    :param mech2: Mechanism with desired names
    :return: Dictionary mapping names from `mech1` to those in `mech2`, and list
        of names from `mech1` that are missing in `mech2`
    """
    match_cols = [Species.amchi, Species.spin, Species.charge]

    # Read in species and names
    spc1_df = mech1.species.rename(col_.to_orig(Species.name))
    spc2_df = mech2.species.select([Species.name, *match_cols])

    # Get names from first mechanism that are included/excluded in second
    incl_spc_df = spc1_df.join(spc2_df, on=match_cols, how="inner")
    excl_spc_df = spc1_df.join(spc2_df, on=match_cols, how="anti")

    orig_col = col_.orig(Species.name)
    name_dct = df_.lookup_dict(incl_spc_df, orig_col, Species.name)
    missing_names = excl_spc_df.get_column(orig_col).to_list()
    return name_dct, missing_names


def network(mech: Mechanism) -> net_.Network:
    """Generate reaction network representation of mechanism.

    :param mech: Mechanism
    :return: Reaction network
    """
    spc_df = mech.species
    rxn_df = mech.reactions

    # Double-check that reagents are sorted
    rxn_df = schema_old.reaction_table_with_sorted_reagents(rxn_df)

    # Add species and reaction indices
    spc_df = df_.with_index(spc_df, net_.Key.id)
    rxn_df = df_.with_index(rxn_df, net_.Key.id)

    # Exluded species
    rgt_names = list(
        itertools.chain(
            *rxn_df[Reaction.reactants].to_list(), *rxn_df[Reaction.products].to_list()
        )
    )
    excl_spc_df = spc_df.filter(~polars.col(Species.name).is_in(rgt_names))

    # Get dataframe of reagents
    rgt_col = "reagents"
    rgt_exprs = [
        rxn_df.select(polars.col(Reaction.reactants).alias(rgt_col), Reaction.formula),
        rxn_df.select(polars.col(Reaction.products).alias(rgt_col), Reaction.formula),
        excl_spc_df.select(
            polars.concat_list(Species.name).alias(rgt_col), Species.formula
        ),
    ]
    rgt_df = polars.concat(rgt_exprs, how="vertical_relaxed").group_by(rgt_col).first()

    # Append species data to reagents dataframe
    names = spc_df[Species.name]
    datas = spc_df.to_struct()
    expr = polars.element().replace_strict(names, datas)
    rgt_df = rgt_df.with_columns(
        polars.col(rgt_col).list.eval(expr).alias(net_.Key.species)
    )

    # Build network object
    def _node_data_from_dict(dct: dict[str, object]):
        key = tuple(dct.get(rgt_col))
        return (key, dct)

    def _edge_data_from_dict(dct: dict[str, object]):
        key1 = tuple(dct.get(Reaction.reactants))
        key2 = tuple(dct.get(Reaction.products))
        return (key1, key2, dct)

    return net_.from_data(
        node_data=list(map(_node_data_from_dict, rgt_df.to_dicts())),
        edge_data=list(map(_edge_data_from_dict, rxn_df.to_dicts())),
    )


def apply_network_function(
    mech: Mechanism, func: Callable, *args, **kwargs
) -> Mechanism:
    """Apply network function to mechanism.

    :param mech: Mechanism
    :param func: Function
    :param *args: Function arguments
    :param **kwargs: Function keyword arguments
    :return: Mechanism
    """
    mech_ = mech.model_copy()

    col_idx = col_.temp()
    mech_.species = df_.with_index(mech_.species, col=col_idx)
    mech_.reactions = df_.with_index(mech_.reactions, col=col_idx)
    net = network(mech_)
    net = func(net, *args, **kwargs)
    spc_idxs = net_.species_values(net, col_idx)
    rxn_idxs = net_.edge_values(net, col_idx)
    mech_.species = mech_.species.filter(polars.col(col_idx).is_in(spc_idxs)).drop(
        col_idx
    )
    mech_.reactions = mech_.reactions.filter(polars.col(col_idx).is_in(rxn_idxs)).drop(
        col_idx
    )
    return mech_


# transformations
def rename(
    mech: Mechanism,
    names: Sequence[str] | Mapping[str, str],
    new_names: Sequence[str] | None = None,
    drop_orig: bool = True,
    drop_missing: bool = False,
) -> Mechanism:
    """Rename species in mechanism.

    :param mech: Mechanism
    :param names: A list of names or mapping from current to new names
    :param new_names: A list of new names
    :param drop_orig: Whether to drop the original names, or include them as `orig`
    :param drop_missing: Whether to drop missing species or keep them
    :return: Mechanism with updated species names
    """
    mech = mech.model_copy()

    if drop_missing:
        mech = with_species(mech, list(names), strict=drop_missing)

    mech.species = spec_table.rename(
        mech.species, names=names, new_names=new_names, drop_orig=drop_orig
    )
    mech.reactions = reac_table.rename(
        mech.reactions, names=names, new_names=new_names, drop_orig=drop_orig
    )
    return mech


def neighborhood(
    mech: Mechanism, species_names: Sequence[str], radius: int = 1
) -> Mechanism:
    """Determine neighborhood of set of species.

    :param mech: Mechanism
    :param species_names: Names of species
    :param radius: Maximum distance of neighbors to include, defaults to 1
    :return: Neighborhood mechanism
    """
    return apply_network_function(
        mech, net_.neighborhood, species_names=species_names, radius=radius
    )


# drop/add reactions
def drop_duplicate_reactions(mech: Mechanism) -> Mechanism:
    """Drop duplicate reactions from mechanism.

    :param mech: Mechanism
    :return: Mechanism without duplicate reactions
    """
    mech = mech.model_copy()

    col_tmp = col_.temp()
    mech.reactions = reac_table.with_key(mech.reactions, col=col_tmp)
    mech.reactions = mech.reactions.unique(col_tmp, maintain_order=True)
    mech.reactions = mech.reactions.drop(col_tmp)
    return mech


def drop_self_reactions(mech: Mechanism) -> Mechanism:
    """Drop self-reactions from mechanism.

    :param mech: Mechanism
    :return: Mechanism
    """
    mech = mech.model_copy()
    mech.reactions = reac_table.drop_self_reactions(mech.reactions)
    return mech


def with_species(
    mech: Mechanism, spc_names: Sequence[str] = (), strict: bool = False
) -> Mechanism:
    """Extract submechanism including species names from list.

    :param mech: Mechanism
    :param spc_names: Names of species to be included
    :param strict: Strictly include these species and no others?
    :return: Submechanism
    """
    return _with_or_without_species(
        mech=mech, spc_names=spc_names, without=False, strict=strict
    )


def without_species(mech: Mechanism, spc_names: Sequence[str] = ()) -> Mechanism:
    """Extract submechanism excluding species names from list.

    :param mech: Mechanism
    :param spc_names: Names of species to be excluded
    :return: Submechanism
    """
    return _with_or_without_species(mech=mech, spc_names=spc_names, without=True)


def _with_or_without_species(
    mech: Mechanism,
    spc_names: Sequence[str] = (),
    without: bool = False,
    strict: bool = False,
) -> Mechanism:
    """Extract submechanism containing or excluding species names from list.

    :param mech: Mechanism
    :param spc_names: Names of species to be included or excluded
    :param without: Extract submechanism *without* these species?
    :param strict: Strictly include these species and no others?
    :return: Submechanism
    """
    # Build appropriate filtering expression
    expr = (
        polars.concat_list(Reaction.reactants, Reaction.products)
        .list.eval(polars.element().is_in(spc_names))
        .list
    )
    expr = expr.all() if strict else expr.any()
    expr = expr.not_() if without else expr

    mech = mech.model_copy()
    mech.reactions = mech.reactions.filter(expr)
    return without_unused_species(mech)


def without_unused_species(mech: Mechanism) -> Mechanism:
    """Remove unused species from mechanism.

    :param mech: Mechanism
    :return: Mechanism without unused species
    """
    mech = mech.model_copy()
    used_names = species_names(mech, rxn_only=True)
    mech.species = mech.species.filter(polars.col(Species.name).is_in(used_names))
    return mech


def with_key(
    mech: Mechanism, col: str = "key", stereo: bool = True
) -> tuple[Mechanism, Mechanism]:
    """Add match key column for species and reactions.

    Currently only accepts a single species key, but could be generalized to accept
    more. The challenge would be in hashing the values.

    :param mech1: First mechanism
    :param spc_key: Species ID column for comparison
    :param col: Output column identifying common species and reactions
    :param stereo: Whether to include stereochemistry
    :return: First and second Mechanisms with intersection columns
    """
    mech = mech.model_copy()
    mech.species = spec_table.with_key(mech.species, col=col, stereo=stereo)
    mech.reactions = reac_table.with_key(
        mech.reactions, col, spc_df=mech.species, stereo=stereo
    )
    return mech


def expand_stereo(
    mech: Mechanism,
    enant: bool = True,
    strained: bool = False,
    distinct_ts: bool = True,
) -> tuple[Mechanism, Mechanism]:
    """Expand stereochemistry for mechanism.

    :param mech: Mechanism
    :param enant: Distinguish between enantiomers?, defaults to True
    :param strained: Include strained stereoisomers?
    :param distinct_ts: Include duplicate reactions for distinct TSs?
    :return: Mechanism with classified reactions, and one with unclassified
    """
    species0 = mech.species
    mech = mech.model_copy()
    err_mech = mech.model_copy()

    # Do species expansion
    mech.species = spec_table.expand_stereo(
        mech.species, enant=enant, strained=strained
    )

    if not reaction_count(mech):
        return mech, mech

    # Add reactant and product AMChIs
    rct_col = Reaction.reactants
    prd_col = Reaction.products
    temp_dct = col_.to_([rct_col, prd_col], col_.temp())
    mech.reactions = reac_table.translate_reagents(
        mech.reactions,
        trans=species0[Species.name],
        trans_into=species0[Species.amchi],
        rct_col=temp_dct.get(rct_col),
        prd_col=temp_dct.get(prd_col),
    )

    # Add "orig" prefix to current reactant and product columns
    orig_dct = col_.to_orig([rct_col, prd_col])
    mech.reactions = mech.reactions.drop(orig_dct.values(), strict=False)
    mech.reactions = mech.reactions.rename(orig_dct)

    # Define expansion function
    name_dct: dict = df_.lookup_dict(
        mech.species, (col_.orig(Species.name), Species.amchi), Species.name
    )

    def _expand_reaction(rchi0s, pchi0s, rname0s, pname0s):
        """Classify reaction and return reaction objects."""
        objs = automol.reac.from_amchis(rchi0s, pchi0s, stereo=False)
        rnames_lst = []
        pnames_lst = []
        ts_amchis = []
        for obj in objs:
            sobjs = automol.reac.expand_stereo(obj, enant=enant, strained=strained)
            for sobj in sobjs:
                # Determine AMChI
                ts_amchi = automol.reac.ts_amchi(sobj)
                # Determine updated equation
                rchis, pchis = automol.reac.amchis(sobj)
                rnames = tuple(map(name_dct.get, zip(rname0s, rchis, strict=True)))
                pnames = tuple(map(name_dct.get, zip(pname0s, pchis, strict=True)))
                if not all(isinstance(n, str) for n in rnames + pnames):
                    return ([], [])

                rnames_lst.append(rnames)
                pnames_lst.append(pnames)
                ts_amchis.append(ts_amchi)
        return rnames_lst, pnames_lst, ts_amchis

    # Do expansion
    cols_in = [*temp_dct.values(), *orig_dct.values()]
    cols_out = (Reaction.reactants, Reaction.products, ReactionStereo.amchi)
    mech.reactions = df_.map_(
        mech.reactions, cols_in, cols_out, _expand_reaction, bar=True
    )

    # Separate out error cases
    err_mech.reactions = mech.reactions.filter(polars.col(rct_col).list.len() == 0)
    mech.reactions = mech.reactions.filter(polars.col(rct_col).list.len() != 0)

    # Expand table by stereoisomers
    err_mech.reactions = err_mech.reactions.drop(
        ReactionStereo.amchi, *orig_dct.keys()
    ).rename(dict(map(reversed, orig_dct.items())))
    mech.reactions = mech.reactions.explode(
        Reaction.reactants, Reaction.products, ReactionStereo.amchi
    )
    mech.reactions = mech.reactions.drop(temp_dct.values())

    if not distinct_ts:
        mech = drop_duplicate_reactions(mech)

    return mech, err_mech


# binary operations
def combine_all(mechs: Sequence[Mechanism]) -> Mechanism:
    """Combine mechanisms into one.

    :param mechs: Mechanisms
    :return: Mechanism
    """
    return functools.reduce(update, mechs)


def intersection(
    mech1: Mechanism, mech2: Mechanism, right: bool = False, stereo: bool = True
) -> tuple[Mechanism, Mechanism]:
    """Determine intersection between one mechanism and another.

    :param mech1: First mechanism
    :param mech2: Second mechanism
    :param right: Whether to return data from `mech2` instead of `mech1`
    :param stereo: Whether to consider stereochemistry
    :return: Mechanism intersection
    """
    col = col_.temp()
    mech1, mech2 = with_intersection_columns(mech1, mech2, col=col, stereo=stereo)
    mech = mech2 if right else mech1
    mech = mech.model_copy()
    mech.reactions = mech.reactions.filter(polars.col(col)).drop(col)
    mech.species = mech.species.filter(polars.col(col)).drop(col)
    return mech


def difference(
    mech1: Mechanism,
    mech2: Mechanism,
    right: bool = False,
    col: str = "intersection",
    stereo: bool = True,
) -> tuple[Mechanism, Mechanism]:
    """Determine difference between one mechanism and another.

    Includes shared species as needed to balance reactions. These can be identified from
    the intersection column, which is named based on the `col` keyword argument.

    :param mech1: First mechanism
    :param mech2: Second mechanism
    :param right: Whether to return data from `mech2` instead of `mech1`
    :param col: Output column identifying common species and reactions
    :param stereo: Whether to consider stereochemistry
    :return: Mechanism difference
    """
    mech1, mech2 = with_intersection_columns(mech1, mech2, col=col, stereo=stereo)
    mech = mech2 if right else mech1
    mech = mech.model_copy()
    mech.reactions = mech.reactions.filter(~polars.col(col)).drop(col)
    # Retain species that are needed to balance reactions
    # (and keep the intersection column, so users can determine which are which)
    spc_names = reac_table.species(mech.reactions)
    mech.species = mech.species.filter(
        ~polars.col(col) | polars.col(Species.name).is_in(spc_names)
    )
    return mech


def update(mech1: Mechanism, mech2: Mechanism, keep_left: bool = False) -> Mechanism:
    """Update one mechanism with species and reactions from another.

    Any overlapping species or reactions will be replaced with those of the second
    mechanism.

    :param mech1: First mechanism
    :param mech2: Second mechanism
    :param keep_left: Whether to keep data for `mech1` instead of `mech2`
    :return: Updated mechanism
    """
    mech1, mech2 = (mech2, mech1) if keep_left else (mech1, mech2)

    # Get intersection information for the first mechanism
    col = col_.temp()
    mech1, _ = with_intersection_columns(mech1, mech2, col=col)
    mech = mech1.model_copy()

    # Determine combined reactions table
    mech.reactions = mech1.reactions.filter(~polars.col(col)).drop(col)
    mech.reactions = polars.concat(
        [mech.reactions, mech2.reactions], how="diagonal_relaxed"
    )

    # Determine combined species table
    mech.species = mech1.species.filter(~polars.col(col)).drop(col)
    mech.species = polars.concat([mech.species, mech2.species], how="diagonal_relaxed")
    return mech


def left_update(
    mech1: Mechanism, mech2: Mechanism, drop_orig: bool = True
) -> Mechanism:
    """Update one mechanism with names and data from another.

    Any overlapping species or reactions will be replaced with those of the second
    mechanism.

    :param mech1: First mechanism
    :param mech2: Second mechanism
    :param drop_orig: Whether to drop the original column values
    :return: Mechanism
    """
    mech = mech1.model_copy()

    col0 = Species.name
    col = col_.prefix(col0, col_.temp())
    mech.species = mech.species.with_columns(polars.col(col0).alias(col))
    mech.species = spec_table.left_update(
        mech.species, mech2.species, drop_orig=drop_orig
    )
    mech.reactions = reac_table.rename(
        mech.reactions, mech.species[col0], mech.species[col], drop_orig=drop_orig
    )
    mech.species = mech.species.drop(col)
    mech.reactions = reac_table.left_update(
        mech.reactions, mech2.reactions, drop_orig=drop_orig
    )
    return mech


def with_intersection_columns(
    mech1: Mechanism, mech2: Mechanism, col: str = "intersection", stereo: bool = True
) -> tuple[Mechanism, Mechanism]:
    """Add columns to Mechanism pair indicating their intersection.

    :param mech1: First mechanism
    :param mech2: Second mechanism
    :param col: Output column identifying common species and reactions
    :param stereo: Whether to consider stereochemistry
    :return: First and second Mechanisms with intersection columns
    """
    mech1 = mech1.model_copy()
    mech2 = mech2.model_copy()

    tmp_col = col_.temp()
    mech1 = with_key(mech1, col=tmp_col, stereo=stereo)
    mech2 = with_key(mech2, col=tmp_col, stereo=stereo)

    # Determine species intersection
    mech1.species, mech2.species = df_.with_intersection_columns(
        mech1.species, mech2.species, comp_col_=tmp_col, col=col
    )

    # Determine reaction intersection
    mech1.reactions, mech2.reactions = df_.with_intersection_columns(
        mech1.reactions, mech2.reactions, comp_col_=tmp_col, col=col
    )

    # Drop temporary columns
    mech1.species = mech1.species.drop(tmp_col)
    mech2.species = mech2.species.drop(tmp_col)
    mech1.reactions = mech1.reactions.drop(tmp_col)
    mech2.reactions = mech2.reactions.drop(tmp_col)

    return mech1, mech2


# parent
def expand_parent_stereo(mech: Mechanism, sub_mech: Mechanism) -> Mechanism:
    """Apply stereoexpansion of submechanism to parent mechanism.

    Produces equivalent of parent mechanism, containing distinct
    stereoisomers of submechanism. Expansion is completely naive, with no
    consideration of stereospecificity, and is simply designed to allow merging of
    stereo-expanded submechanism into parent mechanism.

    :param par_mech: Parent mechanism
    :param sub_mech: Stereo-expanded sub-mechanism
    :return: Equivalent parent mechanism, with distinct stereoisomers from
        sub-mechanism
    """
    mech = mech.model_copy()
    sub_mech = sub_mech.model_copy()

    # 1. Species table
    #   a. Add stereo columns to par_mech species table
    col_dct = col_.to_orig([Species.name, Species.smiles, Species.amchi])
    mech.species = mech.species.rename(col_dct)

    #   b. Group by original names and isolate expanded stereoisomers
    name_col = Species.name
    name_col0 = col_.orig(Species.name)
    sub_mech.species = schema_old.species_table(sub_mech.species, model_=SpeciesStereo)
    sub_mech.species = sub_mech.species.select(*col_dct.keys(), *col_dct.values())
    sub_mech.species = sub_mech.species.group_by(name_col0).agg(polars.all())

    #   c. Form species expansion dictionary, to be used for reaction expansion
    exp_dct: dict[str, list[str]] = df_.lookup_dict(
        sub_mech.species, name_col0, name_col
    )

    #   d. Join on original names, explode, and fill in non-stereoisomer columns
    mech.species = mech.species.join(sub_mech.species, how="left", on=name_col0)
    mech.species = mech.species.drop(polars.selectors.ends_with("_right"))
    mech.species = mech.species.explode(*col_dct.keys())
    mech.species = mech.species.with_columns(
        *(polars.col(k).fill_null(polars.col(v)) for k, v in col_dct.items())
    )

    # 2. Reaction table
    #   a. Identify subset of reactions to be expanded
    has_rate = ReactionRate.rate in mech.reactions
    mech.reactions = reac_table.with_rates(mech.reactions)

    mech.reactions = mech.reactions.with_columns(
        **col_.from_orig([Reaction.reactants, Reaction.products, ReactionRate.rate])
    )
    needs_exp = (
        polars.concat_list(Reaction.reactants, Reaction.products)
        .list.eval(polars.element().is_in(list(exp_dct.keys())))
        .list.any()
    )
    exp_rxn_df = mech.reactions.filter(needs_exp)
    rem_rxn_df = mech.reactions.filter(~needs_exp)

    #   b. Expand and dump to dictionary
    def exp_(rate: autochem.rate.Rate) -> list[dict[str, object]]:
        rates = autochem.rate.expand_lumped(rate, exp_dct=exp_dct)
        return (
            [r.reactants for r in rates],
            [r.products for r in rates],
            [r.rate_constant.model_dump() for r in rates],
        )

    obj_col = col_.temp()
    cols = [Reaction.reactants, Reaction.products, ReactionRate.rate]
    dtypes = list(map(polars.List, map(exp_rxn_df.schema.get, cols)))
    exp_rxn_df = reac_table.with_rate_objects(exp_rxn_df, col=obj_col)
    exp_rxn_df = df_.map_(exp_rxn_df, obj_col, cols, exp_, dtype_=dtypes, bar=True)
    exp_rxn_df = exp_rxn_df.explode(cols)
    mech.reactions = polars.concat([rem_rxn_df, exp_rxn_df.drop(obj_col)])

    if not has_rate:
        mech.reactions = reac_table.without_rates(mech.reactions)
        mech.reactions = mech.reactions.drop(col_.orig(ReactionRate.rate))

    return mech


# building
ReagentValue_ = str | Sequence[str] | None


def enumerate_reactions(
    mech: Mechanism,
    smarts: str,
    rcts_: Sequence[ReagentValue_] | None = None,
    spc_col_: str | Sequence[str] = Species.name,
    src_mech: Mechanism | None = None,
    repeat: int = 1,
    drop_self_rxns: bool = True,
) -> Mechanism:
    """Enumerate reactions for mechanism based on SMARTS reaction template.

    Reactants are listed by position in the SMARTS template. If a sequence of reactants
    is provided, reactions will be enumerated for each of them. If `None` is provided,
    reactions will be enumerated for all species currently in the mechanism.

    :param mech: Mechanism
    :param smarts: SMARTS reaction template
    :param rcts_: Reactants to be used in enumeration (see above)
    :param spc_key_: Species column key(s) for identifying reactants and products
    :param src_mech: Optional source mechanism for species names and data
    :param repeat: Number of times to repeat the enumeration
    :param drop_self_rxns: Whether to drop self-reactions
    :return: Mechanism with enumerated reactions
    """
    for _ in range(repeat):
        mech = _enumerate_reactions(
            mech, smarts, rcts_=rcts_, spc_col_=spc_col_, src_mech=src_mech
        )

    if drop_self_rxns:
        mech = drop_self_reactions(mech)

    return mech


def _enumerate_reactions(
    mech: Mechanism,
    smarts: str,
    rcts_: Sequence[ReagentValue_] | None = None,
    spc_col_: str | Sequence[str] = Species.name,
    src_mech: Mechanism | None = None,
) -> Mechanism:
    """Enumerate reactions for mechanism based on SMARTS reaction template.

    Reactants are listed by position in the SMARTS template. If a sequence of reactants
    is provided, reactions will be enumerated for each of them. If `None` is provided,
    reactions will be enumerated for all species currently in the mechanism.

    :param mech: Mechanism
    :param smarts: SMARTS reaction template
    :param rcts_: Reactants to be used in enumeration (see above)
    :param spc_key_: Species column key(s) for identifying reactants and products
    :param src_mech: Optional source mechanism for species names and data
    :return: Mechanism with enumerated reactions
    """
    # Check reactants argument
    nrcts = automol.smarts.reactant_count(smarts)
    rcts_ = [None] * nrcts if rcts_ is None else rcts_
    assert len(rcts_) == nrcts, f"Reactant count mismatch for {smarts}:\n{rcts_}"

    # Process reactants argument
    mech = mech.model_copy()
    spc_pool = df_.values(mech.species, spc_col_)
    rcts_ = [spc_pool if r is None else [r] if isinstance(r, str) else r for r in rcts_]

    # Enumerate reactions
    rxn_spc_ids = []
    for rcts in itertools.product(*rcts_):
        rct_spc_ids = spec_table.species_ids(
            mech.species, rcts, col_=spc_col_, try_fill=True
        )
        rct_chis, *_ = zip(*rct_spc_ids, strict=True)
        for rxn in automol.reac.enum.from_amchis(smarts, rct_chis):
            _, prd_chis = automol.reac.amchis(rxn)
            prd_spc_ids = spec_table.species_ids(
                mech.species, prd_chis, col_=Species.amchi, try_fill=True
            )
            rxn_spc_ids.append((rct_spc_ids, prd_spc_ids))

    # Form the updated species DataFrame
    spc_ids = list(itertools.chain.from_iterable(r + p for r, p in rxn_spc_ids))
    spc_ids = list(mit.unique_everseen(spc_ids))
    mech.species = spec_table.add_missing_species_by_id(mech.species, spc_ids)
    mech.species = (
        mech.species
        if src_mech is None
        else spec_table.left_update(mech.species, src_mech.species)
    )

    # Form the updated reactions DataFrame
    spc_names = spec_table.species_names_by_id(mech.species, spc_ids)
    name_ = dict(zip(spc_ids, spc_names, strict=True)).get
    rxn_ids = [[list(map(name_, r)) for r in rs] for rs in rxn_spc_ids]
    rxn_ids = list(mit.unique_everseen(rxn_ids))
    mech.reactions = reac_table.add_missing_reactions_by_id(mech.reactions, rxn_ids)
    mech = mech if src_mech is None else left_update(mech, src_mech)
    return drop_duplicate_reactions(mech)


# sorting
def with_sort_data(mech: Mechanism) -> Mechanism:
    """Add columns to sort mechanism by species and reactions.

    :param mech: Mechanism
    :return: Mechanism with sort columns
    """
    mech = mech.model_copy()

    # Sort species by formula
    mech.species = spec_table.sort_by_formula(mech.species)

    # Sort reactions by shape and by reagent names
    idx_col = col_.temp()
    mech.reactions = mech.reactions.sort(
        polars.col(Reaction.reactants).list.len(),
        polars.col(Reaction.products).list.len(),
        df_.list_to_struct_expression(mech.reactions, Reaction.reactants),
        df_.list_to_struct_expression(mech.reactions, Reaction.products),
    )
    mech.reactions = df_.with_index(mech.reactions, idx_col)

    # Generate sort data from network
    srt_dct = net_.sort_data(network(mech), idx_col)
    srt_data = [
        {
            idx_col: i,
            ReactionSorted.pes: p,
            ReactionSorted.subpes: s,
            ReactionSorted.channel: c,
        }
        for i, (p, s, c) in srt_dct.items()
    ]
    srt_schema = {idx_col: polars.UInt32, **schema_old.types([ReactionSorted])}
    srt_df = polars.DataFrame(srt_data, schema=srt_schema)

    # Add sort data to reactions dataframe and sort
    mech.reactions = mech.reactions.join(srt_df, on=idx_col, how="left")
    mech.reactions = mech.reactions.drop(idx_col)
    mech.reactions = mech.reactions.sort(
        ReactionSorted.pes, ReactionSorted.subpes, ReactionSorted.channel
    )
    return mech


# comparison
def are_equivalent(mech1: Mechanism, mech2: Mechanism) -> bool:
    """Determine whether two mechanisms are equivalent.

    (Currently too strict -- need to figure out how to handle nested float comparisons
    in Struct columns.)

    Waiting on:
     - https://github.com/pola-rs/polars/issues/11067 (to be used with .unnest())
    and/or:
     - https://github.com/pola-rs/polars/issues/18936

    :param mech1: First mechanism
    :param mech2: Second mechanism
    :return: `True` if they are, `False` if they aren't
    """
    same_reactions = mech1.reactions.equals(mech2.reactions)
    same_species = mech1.species.equals(mech2.species)
    return same_reactions and same_species


# read/write
def string(mech: Mechanism) -> str:
    """Write mechanism to JSON string.

    :param mech: Mechanism
    :return: Mechanism JSON string
    """
    return mech.model_dump_json()


def from_string(mech_str: str) -> Mechanism:
    """Read mechanism from JSON string.

    :param mech_str: Mechanism JSON string
    :return: Mechanism
    """
    return Mechanism.model_validate_json(mech_str)


# display
def display(
    mech: Mechanism,
    stereo: bool = True,
    color_subpes: bool = True,
    species_centered: bool = False,
    exclude_formulas: Sequence[str] = net_.DEFAULT_EXCLUDE_FORMULAS,
    height: str = "750px",
    out_name: str = "net.html",
    out_dir: str = ".automech",
    open_browser: bool = True,
) -> None:
    """Display mechanism as reaction network.

    :param mech: Mechanism
    :param stereo: Include stereochemistry in species drawings?, defaults to True
    :param color_subpes: Add distinct colors to different PESs
    :param species_centered: Display as species-centered network?
    :param exclude_formulas: If species-centered, exclude these species from display
    :param height: Control height of frame
    :param out_name: Name of HTML file for network visualization
    :param out_dir: Name of directory for saving network visualization
    :param open_browser: Whether to open browser automatically
    """
    net_.display(
        net=network(mech),
        stereo=stereo,
        color_subpes=color_subpes,
        species_centered=species_centered,
        exclude_formulas=exclude_formulas,
        height=height,
        out_name=out_name,
        out_dir=out_dir,
        open_browser=open_browser,
    )


def display_species(
    mech: Mechanism,
    spc_vals_: Sequence[str] | None = None,
    spc_key_: str | Sequence[str] = Species.name,
    stereo: bool = True,
    keys: tuple[str, ...] = (
        Species.name,
        Species.smiles,
    ),
):
    """Display species in mechanism.

    :param mech: Mechanism
    :param vals_: Species column value(s) list for selection
    :param spc_key_: Species column key(s) for selection
    :param stereo: Include stereochemistry in species drawings?, defaults to True
    :param keys: Keys of extra columns to print
    """
    # Read in mechanism data
    spc_df = mech.species

    if spc_vals_ is not None:
        spc_df = spec_table.filter(spc_df, vals_=spc_vals_, col_=spc_key_)
        id_ = [spc_key_] if isinstance(spc_key_, str) else spc_key_
        keys = [*id_, *(k for k in keys if k not in id_)]

    def _display_species(chi, *vals):
        """Display a species."""
        # Print requested information
        for key, val in zip(keys, vals, strict=True):
            print(f"{key}: {val}")

        automol.amchi.display(chi, stereo=stereo)

    # Display requested reactions
    df_.map_(spc_df, (Species.amchi, *keys), None, _display_species)


def display_reactions(
    mech: Mechanism,
    eqs: Collection | None = None,
    stereo: bool = True,
    keys: Sequence[str] = (),
    spc_keys: Sequence[str] = (Species.smiles,),
):
    """Display reactions in mechanism.

    :param mech: Mechanism
    :param eqs: Optionally, specify specific equations to visualize
    :param stereo: Include stereochemistry in species drawings?, defaults to True
    :param keys: Keys of extra columns to print
    :param spc_keys: Optionally, translate reactant and product names into these
        species dataframe values
    """
    # Read in mechanism data
    spc_df = mech.species
    rxn_df = mech.reactions

    chi_dct = df_.lookup_dict(spc_df, Species.name, Species.amchi)
    trans_dcts = {k: df_.lookup_dict(spc_df, Species.name, k) for k in spc_keys}

    obj_col, eq_col = col_.temp(), col_.temp()
    rxn_df = reac_table.with_rate_objects(rxn_df, col=obj_col)
    rxn_df = rxn_df.with_columns(
        polars.col(obj_col)
        .map_elements(autochem.rate.chemkin_equation, return_dtype=polars.String)
        .alias(eq_col)
    )

    if eqs is not None:
        # eq_df = polars.DataFrame({})
        eqs = list(map(data.reac.standardize_chemkin_equation, eqs))
        rxn_df = rxn_df.filter(polars.col("eq").is_in(eqs))

    def _display_reaction(eq, *vals):
        """Add a node to network."""
        # Print requested information
        for key, val in zip(keys, vals, strict=True):
            print(f"{key}: {val}")

        # Display reaction
        rchis, pchis, *_ = data.reac.read_chemkin_equation(eq, trans_dct=chi_dct)

        for key, trans_dct in trans_dcts.items():
            rvals, pvals, *_ = data.reac.read_chemkin_equation(eq, trans_dct=trans_dct)
            print(f"Species `name`=>`{key}` translation")
            print(f"  reactants = {rvals}")
            print(f"  products = {pvals}")

        if not all(isinstance(n, str) for n in rchis + pchis):
            print(f"Some ChIs missing from species table: {rchis} = {pchis}")
        else:
            automol.amchi.display_reaction(rchis, pchis, stereo=stereo)

    # Display requested reactions
    df_.map_(rxn_df, ("eq", *keys), None, _display_reaction)
