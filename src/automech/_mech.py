"""Definition and core functionality of mechanism data structure."""

import dataclasses
import functools
import itertools
import json
import textwrap
from collections.abc import Callable, Collection, Mapping, Sequence

import automol
import more_itertools as mit
import polars
import pydantic

from . import data, reac_table, schema, spec_table
from . import net as net_
from .schema import (
    Model,
    Reaction,
    ReactionRateOld,
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
    species: DataFrame_
    thermo_temps: tuple[float, float, float] | None = None


@dataclasses.dataclass
class OldMechanism:
    """Chemical kinetic mechanism."""

    reactions: polars.DataFrame
    species: polars.DataFrame
    rate_units: tuple[str, str] | None = None
    thermo_temps: tuple[float, float, float] | None = None

    def __post_init__(self):
        """Initialize attributes."""
        if self.thermo_temps is not None:
            assert len(self.thermo_temps) == 3, f"Bad thermo_temps: {self.thermo_temps}"
            self.thermo_temps = tuple(map(float, self.thermo_temps))

        if self.rate_units is not None:
            assert len(self.rate_units) == 2, f"Bad rate_units: {self.rate_units}"
            self.rate_units = tuple(map(str, self.rate_units))

        if not isinstance(self.reactions, polars.DataFrame):
            self.reactions = polars.DataFrame(self.reactions, infer_schema_length=None)

        if not isinstance(self.species, polars.DataFrame):
            self.species = polars.DataFrame(self.species, infer_schema_length=None)

        self.species = schema.species_table(self.species)
        self.reactions, _ = schema.reaction_table_with_errors(
            self.reactions, spc_df=self.species
        )

    def __repr__(self):
        rxn_df_rep = textwrap.indent(repr(self.reactions), "  ")
        spc_df_rep = textwrap.indent(repr(self.species), "  ")
        attrib_strs = [
            f"reactions=DataFrame(\n{rxn_df_rep}\n)",
            f"species=DataFrame(\n{spc_df_rep}\n)",
            f"rate_units={self.rate_units}",
            f"thermo_temps={self.thermo_temps}",
        ]
        attrib_strs = [textwrap.indent(s, "  ") for s in attrib_strs]
        attrib_str = ",\n".join(attrib_strs)
        return f"Mechanism(\n{attrib_str},\n)"

    def __iter__(self):
        """Iterate over mechanism's dictionary representation."""
        rxn_dct = json.loads(self.reactions.write_json())
        spc_dct = json.loads(self.species.write_json())
        mech_dct = {**self.__dict__, "reactions": rxn_dct, "species": spc_dct}
        yield from mech_dct.items()


# constructors
def from_data(
    rxn_inp: str | polars.DataFrame | None = None,
    spc_inp: str | polars.DataFrame | None = None,
    thermo_temps: tuple[float, float, float] | None = None,
    rate_units: tuple[str, str] | None = None,
    rxn_models: Sequence[Model] = (),
    spc_models: Sequence[Model] = (),
    fail_on_error: bool = True,
) -> OldMechanism:
    """Construct mechanism object from data.

    :param rxn_inp: Reactions table, as CSV file path or DataFrame
    :param spc_inp: Species table, as CSV file path or DataFrame
    :param rxn_models: Extra reaction models to validate against
    :param spc_models: Extra species models to validate against
    :param fail_on_error: Whether to raise exception if there is inconsistency
    :return: Mechanism object
    """
    spc_inp = (
        polars.DataFrame([], schema=schema.types(Species))
        if spc_inp is None
        else spc_inp
    )
    rxn_inp = (
        polars.DataFrame([], schema=schema.types(Species))
        if rxn_inp is None
        else rxn_inp
    )

    spc_df = spc_inp if isinstance(spc_inp, polars.DataFrame) else df_.from_csv(spc_inp)
    rxn_df = rxn_inp if isinstance(rxn_inp, polars.DataFrame) else df_.from_csv(rxn_inp)
    spc_df = schema.species_table(spc_df, model_=spc_models)
    rxn_df, _ = schema.reaction_table_with_errors(
        rxn_df, model_=rxn_models, spc_df=spc_df, fail_on_error=fail_on_error
    )
    mech = OldMechanism(
        reactions=rxn_df,
        species=spc_df,
        thermo_temps=thermo_temps,
        rate_units=rate_units,
    )
    return mech


def from_network(net: net_.Network) -> OldMechanism:
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
    return from_data(rxn_inp=rxn_df, spc_inp=spc_df, fail_on_error=False)


def from_smiles(
    spc_smis: Sequence[str] = (),
    rxn_smis: Sequence[str] = (),
    name_dct: dict[str, str] | None = None,
    spin_dct: dict[str, int] | None = None,
    charge_dct: dict[str, int] | None = None,
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
    name_dct = {} if name_dct is None else name_dct
    spin_dct = {} if spin_dct is None else spin_dct
    charge_dct = {} if charge_dct is None else charge_dct

    # Add in any missing species from reaction SMILES
    spc_smis_by_rxn = [
        rs + ps
        for (rs, ps) in map(automol.smiles.reaction_reactants_and_products, rxn_smis)
    ]
    spc_smis = list(mit.unique_everseen(itertools.chain(spc_smis, *spc_smis_by_rxn)))

    # Build species dataframe
    chis = list(map(automol.smiles.amchi, spc_smis))
    chi_dct = dict(zip(spc_smis, chis, strict=True))
    name_dct = {chi_dct[k]: v for k, v in name_dct.items() if k in spc_smis}
    spin_dct = {chi_dct[k]: v for k, v in spin_dct.items() if k in spc_smis}
    charge_dct = {chi_dct[k]: v for k, v in charge_dct.items() if k in spc_smis}
    data_dct = {Species.smiles: spc_smis, Species.amchi: chis}
    dt = schema.species_types(data_dct.keys())
    spc_df = polars.DataFrame(data=data_dct, schema=dt)
    spc_df = schema.species_table(
        spc_df, name_dct=name_dct, spin_dct=spin_dct, charge_dct=charge_dct
    )

    # Left-update by species key, if source mechanism was provided
    if src_mech is not None:
        spc_df = spec_table.left_update(spc_df, species(src_mech), drop_orig=True)

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
    dt = schema.reaction_types([Reaction.reactants, Reaction.products])
    rxn_df = polars.DataFrame(data=data_lst, schema=dt)

    mech = Mechanism(reactions=rxn_df, species=spc_df)
    return mech if src_mech is None else left_update(mech, src_mech)


# getters
def species(mech: OldMechanism) -> polars.DataFrame:
    """Get species DataFrame for mechanism.

    :param mech: Mechanism
    :return: Mechanism's species DataFrame
    """
    return mech.species


species_ = species


def reactions(mech: OldMechanism) -> polars.DataFrame:
    """Get reactions DataFrame for mechanism.

    :param mech: Mechanism
    :return: Mechanism's reactions DataFrame
    """
    return mech.reactions


reactions_ = reactions


def thermo_temperatures(mech: OldMechanism) -> tuple[float, float, float] | None:
    """Get thermo temperatures for mechanism.

    :param mech: Mechanism
    :return: Thermo temperatures
    """
    return mech.thermo_temps


thermo_temperatures_ = thermo_temperatures


def rate_units(mech_: OldMechanism | Sequence[OldMechanism]) -> tuple[str, str] | None:
    """Get rate units for mechanism(s).

    If multiple mechanisms are passed in, this gets the first non-null units.

    :param mech_: Mechanism(s)
    :return: Rate units
    """
    if isinstance(mech_, Sequence):
        return next((rate_units(m) for m in mech_ if m.rate_units is not None), None)
    return mech_.rate_units


rate_units_ = rate_units


# setters
def set_species(mech: OldMechanism, spc_df: polars.DataFrame) -> OldMechanism:
    """Set species DataFrame for mechanism.

    :param mech: Mechanism
    :param spc_df: New species DataFrame
    :return: Mechanism with updated species
    """
    return from_data(
        rxn_inp=reactions(mech),
        spc_inp=spc_df,
        thermo_temps=thermo_temperatures(mech),
        rate_units=rate_units(mech),
    )


def set_reactions(mech: OldMechanism, rxn_df: polars.DataFrame) -> OldMechanism:
    """Set reactions DataFrame for mechanism.

    :param mech: Mechanism
    :param rxn_df: New reactions DataFrame
    :return: Mechanism with updated reactions
    """
    return from_data(
        rxn_inp=rxn_df,
        spc_inp=species(mech),
        thermo_temps=thermo_temperatures(mech),
        rate_units=rate_units(mech),
    )


def set_thermo_temperatures(
    mech: OldMechanism, temps: tuple[float, float, float] | None
) -> OldMechanism:
    """Set thermo temperatures for mechanism.

    :param mech: Mechanism
    :param temps: New thermo temperatures
    :return: Mechanism with updated thermo temperatures
    """
    return from_data(
        rxn_inp=reactions(mech),
        spc_inp=species(mech),
        thermo_temps=temps,
        rate_units=rate_units(mech),
    )


def set_rate_units(
    mech: OldMechanism, units: tuple[str, str] | None, scale_rates: bool = True
) -> OldMechanism:
    """Set rate units for mechanism.

    :param mech: Mechanism
    :param units: New rate units
    :param scale_rates: Scale rates if changing units?
    :return: Mechanism with updated rate units
    """
    rxn_df = reactions(mech)
    units0 = rate_units(mech)
    if scale_rates and reac_table.has_rates(rxn_df):
        e_unit0, a_unit0 = map(str.lower, units0)
        e_unit, a_unit = map(str.lower, units)
        assert (
            a_unit == a_unit0
        ), f"{a_unit} != {a_unit0} (A conversion not yet implemented)"

        def _convert(rate_dct):
            rate_obj = data.rate.from_data(**rate_dct)
            rate_obj = data.rate.convert_energy_units(rate_obj, e_unit0, e_unit)
            return dict(rate_obj)

        if e_unit0 != e_unit:
            rxn_df = df_.map_(
                rxn_df, ReactionRateOld.rate, ReactionRateOld.rate, _convert
            )

    return from_data(
        rxn_inp=rxn_df,
        spc_inp=species(mech),
        thermo_temps=thermo_temperatures(mech),
        rate_units=units,
    )


def update_data(
    mech: OldMechanism,
    rxn_df: polars.DataFrame | None = None,
    spc_df: polars.DataFrame | None = None,
    thermo_temps: tuple[float, float, float] | None = None,
    rate_units: tuple[str, str] | None = None,
) -> OldMechanism:
    """Update mechanism data.

    :param rxn_df: Reactions DataFrame
    :param spc_df: Species DataFrame
    :param thermo_temps: Thermodynamic temperatures
    :param rate_units: Rate units
    :return: Mechanism
    """
    return from_data(
        rxn_inp=reactions(mech) if rxn_df is None else rxn_df,
        spc_inp=species(mech) if spc_df is None else spc_df,
        thermo_temps=(
            thermo_temperatures(mech) if thermo_temps is None else thermo_temps
        ),
        rate_units=rate_units_(mech) if rate_units is None else rate_units,
    )


# properties
def species_count(mech: OldMechanism) -> int:
    """Get number of species in mechanism.

    :param mech: Mechanism
    :return: Number of species
    """
    return df_.count(species(mech))


def reaction_count(mech: OldMechanism) -> int:
    """Get number of reactions in mechanism.

    :param mech: Mechanism
    :return: Number of reactions
    """
    return df_.count(reactions(mech))


def reagents(mech: OldMechanism) -> list[list[str]]:
    """Get sets of reagents in mechanism.

    :param mech: Mechanism
    :return: Sets of reagents
    """
    return reac_table.reagents(reactions(mech))


def species_names(
    mech: OldMechanism,
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

    spc_df = species(mech)

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
        rxn_df = reactions(mech)
        rxn_spc_names = reac_table.species(rxn_df)
        spc_names = [n for n in spc_names if n in rxn_spc_names]

    return spc_names


def reaction_reactants(mech: OldMechanism) -> list[list[str]]:
    """Get reactants of reactions in mechanism.

    :param mech: Mechanism
    :return: Reaction reactants
    """
    rxn_df = reactions(mech)
    return rxn_df[Reaction.reactants].to_list()


def reaction_products(mech: OldMechanism) -> list[list[str]]:
    """Get products of reactions in mechanism.

    :param mech: Mechanism
    :return: Reaction products
    """
    rxn_df = reactions(mech)
    return rxn_df[Reaction.products].to_list()


def reaction_reactants_and_products(
    mech: OldMechanism,
) -> list[tuple[list[str], list[str]]]:
    """Get reactants and products of reactions in mechanism.

    :param mech: Mechanism
    :return: Reaction reactants and products
    """
    rxn_df = reactions(mech)
    return rxn_df[[Reaction.reactants, Reaction.products]].rows()


def reaction_equations(mech: OldMechanism) -> list[str]:
    """Get equations of reactions in mechanism.

    :param mech: Mechanism
    :return: Reaction equations
    """
    rps = reaction_reactants_and_products(mech)
    return list(itertools.starmap(data.reac.write_chemkin_equation, rps))


def reaction_species_names(mech: OldMechanism) -> list[str]:
    """Get names of all species that participate in reactions.

    :param mech: Mechanism
    :return: Reaction species names
    """
    eqs = reaction_equations(mech)
    rxn_names = [r + p for r, p, *_ in map(data.reac.read_chemkin_equation, eqs)]
    return list(mit.unique_everseen(itertools.chain(*rxn_names)))


def rename_dict(
    mech1: OldMechanism, mech2: OldMechanism
) -> tuple[dict[str, str], list[str]]:
    """Generate dictionary for renaming species names from one mechanism to another.

    :param mech1: Mechanism with original names
    :param mech2: Mechanism with desired names
    :return: Dictionary mapping names from `mech1` to those in `mech2`, and list
        of names from `mech1` that are missing in `mech2`
    """
    match_cols = [Species.amchi, Species.spin, Species.charge]

    # Read in species and names
    spc1_df = species(mech1)
    spc1_df = spc1_df.rename(col_.to_orig(Species.name))

    spc2_df = species(mech2)
    spc2_df = spc2_df.select([Species.name, *match_cols])

    # Get names from first mechanism that are included/excluded in second
    incl_spc_df = spc1_df.join(spc2_df, on=match_cols, how="inner")
    excl_spc_df = spc1_df.join(spc2_df, on=match_cols, how="anti")

    orig_col = col_.orig(Species.name)
    name_dct = df_.lookup_dict(incl_spc_df, orig_col, Species.name)
    missing_names = excl_spc_df[orig_col].to_list()
    return name_dct, missing_names


def network(mech: OldMechanism) -> net_.Network:
    """Generate reaction network representation of mechanism.

    :param mech: Mechanism
    :return: Reaction network
    """
    spc_df = species(mech)
    rxn_df = reactions(mech)

    # Double-check that reagents are sorted
    rxn_df = schema.reaction_table_with_sorted_reagents(rxn_df)

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
    mech: OldMechanism, func: Callable, *args, **kwargs
) -> OldMechanism:
    """Apply network function to mechanism.

    :param mech: Mechanism
    :param func: Function
    :param *args: Function arguments
    :param **kwargs: Function keyword arguments
    :return: Mechanism
    """
    mech0 = mech

    col_idx = col_.temp()
    spc_df = df_.with_index(species(mech0), col=col_idx)
    rxn_df = df_.with_index(reactions(mech0), col=col_idx)
    mech0 = update_data(mech0, rxn_df=rxn_df, spc_df=spc_df)
    net0 = network(mech0)
    net = func(net0, *args, **kwargs)
    spc_idxs = net_.species_values(net, col_idx)
    rxn_idxs = net_.edge_values(net, col_idx)
    spc_df = spc_df.filter(polars.col(col_idx).is_in(spc_idxs)).drop(col_idx)
    rxn_df = rxn_df.filter(polars.col(col_idx).is_in(rxn_idxs)).drop(col_idx)
    return update_data(mech0, rxn_df=rxn_df, spc_df=spc_df)


# transformations
def rename(
    mech: OldMechanism,
    names: Sequence[str] | Mapping[str, str],
    new_names: Sequence[str] | None = None,
    drop_orig: bool = True,
    drop_missing: bool = False,
) -> OldMechanism:
    """Rename species in mechanism.

    :param mech: Mechanism
    :param names: A list of names or mapping from current to new names
    :param new_names: A list of new names
    :param drop_orig: Whether to drop the original names, or include them as `orig`
    :param drop_missing: Whether to drop missing species or keep them
    :return: Mechanism with updated species names
    """
    if drop_missing:
        mech = with_species(mech, list(names), strict=drop_missing)

    spc_df = spec_table.rename(
        species(mech), names=names, new_names=new_names, drop_orig=drop_orig
    )
    rxn_df = reac_table.rename(
        reactions(mech), names=names, new_names=new_names, drop_orig=drop_orig
    )
    return update_data(mech, rxn_df=rxn_df, spc_df=spc_df)


def remove_all_reactions(mech: OldMechanism) -> OldMechanism:
    """Clear reactions from mechanism.

    :param mech: Mechanism
    :return: Mechanism without reactions
    """
    return set_reactions(mech, reactions(mech).clear())


def add_reactions(mech: OldMechanism, rxn_df: polars.DataFrame) -> OldMechanism:
    """Add reactions from DataFrame to mechanism.

    :param mech: Mechanism
    :param rxn_df: Reactions DataFrame
    :return: Mechanism with added reactions
    """
    rxn_df0 = reactions(mech)
    return set_reactions(mech, polars.concat([rxn_df0, rxn_df], how="diagonal_relaxed"))


def select_pes(
    mech: OldMechanism,
    formula_: str | dict | Sequence[str | dict],
    exclude: bool = False,
) -> OldMechanism:
    """Select (or exclude) PES by formula(s).

    :param mech: Mechanism
    :param formula_: PES formula(s) to include or exclude
    :param exclude: Whether to exclude or include the formula(s)
    :return: Mechanism
    """
    rxn_df = reac_table.select_pes(reactions(mech), formula_, exclude=exclude)
    return without_unused_species(set_reactions(mech, rxn_df))


def neighborhood(
    mech: OldMechanism, species_names: Sequence[str], radius: int = 1
) -> OldMechanism:
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
def drop_duplicate_reactions(mech: OldMechanism) -> OldMechanism:
    """Drop duplicate reactions from mechanism.

    :param mech: Mechanism
    :return: Mechanism without duplicate reactions
    """
    col_tmp = col_.temp()
    rxn_df = reactions(mech)
    rxn_df = reac_table.with_key(rxn_df, col=col_tmp)
    rxn_df = rxn_df.unique(col_tmp, maintain_order=True)
    rxn_df = rxn_df.drop(col_tmp)
    return set_reactions(mech, rxn_df)


def drop_self_reactions(mech: OldMechanism) -> OldMechanism:
    """Drop self-reactions from mechanism.

    :param mech: Mechanism
    :return: Mechanism
    """
    rxn_df = reac_table.drop_self_reactions(reactions(mech))
    return set_reactions(mech, rxn_df)


def with_species(
    mech: OldMechanism, spc_names: Sequence[str] = (), strict: bool = False
) -> OldMechanism:
    """Extract submechanism including species names from list.

    :param mech: Mechanism
    :param spc_names: Names of species to be included
    :param strict: Strictly include these species and no others?
    :return: Submechanism
    """
    return _with_or_without_species(
        mech=mech, spc_names=spc_names, without=False, strict=strict
    )


def without_species(mech: OldMechanism, spc_names: Sequence[str] = ()) -> OldMechanism:
    """Extract submechanism excluding species names from list.

    :param mech: Mechanism
    :param spc_names: Names of species to be excluded
    :return: Submechanism
    """
    return _with_or_without_species(mech=mech, spc_names=spc_names, without=True)


def _with_or_without_species(
    mech: OldMechanism,
    spc_names: Sequence[str] = (),
    without: bool = False,
    strict: bool = False,
) -> OldMechanism:
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

    rxn_df = reactions(mech)

    rxn_df = rxn_df.filter(expr)
    return without_unused_species(
        from_data(
            rxn_inp=rxn_df,
            spc_inp=species(mech),
            thermo_temps=thermo_temperatures(mech),
            rate_units=rate_units(mech),
        )
    )


def without_unused_species(mech: OldMechanism) -> OldMechanism:
    """Remove unused species from mechanism.

    :param mech: Mechanism
    :return: Mechanism without unused species
    """
    spc_df = species(mech)
    used_names = species_names(mech, rxn_only=True)
    spc_df = spc_df.filter(polars.col(Species.name).is_in(used_names))
    return set_species(mech, spc_df)


def with_rates(mech: OldMechanism) -> OldMechanism:
    """Add dummy placeholder rates to this Mechanism, if missing.

    This is mainly needed for ChemKin mechanism writing.

    :param rxn_df: Mechanism
    :return: Mechanism with dummy rates, if missing
    """
    mech = mech.model_copy()
    mech.reactions = reac_table.with_rates(mech.reactions)
    return mech


def with_key(
    mech: OldMechanism, col: str = "key", stereo: bool = True
) -> tuple[OldMechanism, OldMechanism]:
    """Add match key column for species and reactions.

    Currently only accepts a single species key, but could be generalized to accept
    more. The challenge would be in hashing the values.

    :param mech1: First mechanism
    :param spc_key: Species ID column for comparison
    :param col: Output column identifying common species and reactions
    :param stereo: Whether to include stereochemistry
    :return: First and second Mechanisms with intersection columns
    """
    spc_df = spec_table.with_key(species(mech), col=col, stereo=stereo)
    rxn_df = reac_table.with_key(reactions(mech), col, spc_df=spc_df, stereo=stereo)
    return update_data(mech, rxn_df=rxn_df, spc_df=spc_df)


def expand_stereo(
    mech: OldMechanism,
    enant: bool = True,
    strained: bool = False,
    distinct_ts: bool = True,
) -> tuple[OldMechanism, OldMechanism]:
    """Expand stereochemistry for mechanism.

    :param mech: Mechanism
    :param enant: Distinguish between enantiomers?, defaults to True
    :param strained: Include strained stereoisomers?
    :param distinct_ts: Include duplicate reactions for distinct TSs?
    :return: Mechanism with classified reactions, and one with unclassified
    """
    # Read in mechanism data
    spc_df0: polars.DataFrame = species(mech)
    rxn_df: polars.DataFrame = reactions(mech)

    # Do species expansion
    spc_df = _expand_species_stereo(spc_df0, enant=enant, strained=strained)

    if not reaction_count(mech):
        mech = set_species(mech, spc_df)
        return mech, mech

    # Add reactant and product AMChIs
    rxn_df = reac_table.translate_reagents(
        rxn_df,
        trans=spc_df0[Species.name],
        trans_into=spc_df0[Species.amchi],
        rct_col="ramchis",
        prd_col="pamchis",
    )

    # Add "orig" prefix to current reactant and product columns
    col_dct = col_.to_orig([Reaction.reactants, Reaction.products])
    rxn_df = rxn_df.drop(col_dct.values(), strict=False)
    rxn_df = rxn_df.rename(col_dct)

    # Define expansion function
    name_dct: dict = df_.lookup_dict(
        spc_df, (col_.orig(Species.name), Species.amchi), Species.name
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
    cols_in = (
        "ramchis",
        "pamchis",
        *map(col_.orig, [Reaction.reactants, Reaction.products]),
    )
    cols_out = (Reaction.reactants, Reaction.products, ReactionStereo.amchi)
    rxn_df = df_.map_(rxn_df, cols_in, cols_out, _expand_reaction, bar=True)

    # Separate out error cases
    err_df = rxn_df.filter(polars.col(Reaction.reactants).list.len() == 0)
    rxn_df = rxn_df.filter(polars.col(Reaction.reactants).list.len() != 0)

    # Expand table by stereoisomers
    err_df = err_df.drop(ReactionStereo.amchi, *col_dct.keys()).rename(
        dict(map(reversed, col_dct.items()))
    )
    rxn_df = rxn_df.explode(Reaction.reactants, Reaction.products, ReactionStereo.amchi)
    rxn_df = rxn_df.drop("ramchis", "pamchis")

    # Form new mechanisms
    mech = from_data(
        rxn_df,
        spc_df,
        thermo_temps=thermo_temperatures(mech),
        rate_units=rate_units(mech),
    )
    err_mech = from_data(
        err_df,
        spc_df0,
        thermo_temps=thermo_temperatures(mech),
        rate_units=rate_units(mech),
    )

    if not distinct_ts:
        mech = drop_duplicate_reactions(mech)

    return mech, err_mech


def _expand_species_stereo(
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
        return automol.amchi.expand_stereo(chi, enant=enant)

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


# binary operations
def common_rate_units_all(mechs: Sequence[OldMechanism]) -> list[OldMechanism]:
    """Convert mechanisms to common rate units.

    :param mechs: Mechanisms
    :return: Mechanisms with common rate units
    """
    units = rate_units(mechs)
    return [set_rate_units(m, units=units) for m in mechs]


def combine_all(mechs: Sequence[OldMechanism]) -> OldMechanism:
    """Combine mechanisms into one.

    :param mechs: Mechanisms
    :return: Mechanism
    """
    mechs = common_rate_units_all(mechs)
    return functools.reduce(update, mechs)


def intersection(
    mech1: OldMechanism, mech2: OldMechanism, right: bool = False, stereo: bool = True
) -> tuple[OldMechanism, OldMechanism]:
    """Determine intersection between one mechanism and another.

    :param mech1: First mechanism
    :param mech2: Second mechanism
    :param right: Whether to return data from `mech2` instead of `mech1`
    :param stereo: Whether to consider stereochemistry
    :return: Mechanism intersection
    """
    tmp_col = col_.temp()
    mech1, mech2 = with_intersection_columns(mech1, mech2, col=tmp_col, stereo=stereo)
    mech = mech2 if right else mech1
    rxn_df = reactions(mech).filter(polars.col(tmp_col)).drop(tmp_col)
    spc_df = species(mech).filter(polars.col(tmp_col)).drop(tmp_col)
    return update_data(mech, rxn_df=rxn_df, spc_df=spc_df)


def difference(
    mech1: OldMechanism,
    mech2: OldMechanism,
    right: bool = False,
    col: str = "intersection",
    stereo: bool = True,
) -> tuple[OldMechanism, OldMechanism]:
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
    rxn_df = reactions(mech).filter(~polars.col(col)).drop(col)
    # Retain species that are needed to balance reactions
    # (and keep the intersection column, so users can determine which are which)
    rxn_spcs = reac_table.species(rxn_df)
    spc_df = species(mech).filter(
        ~polars.col(col) | polars.col(Species.name).is_in(rxn_spcs)
    )
    return update_data(mech, rxn_df=rxn_df, spc_df=spc_df)


def update(
    mech1: OldMechanism, mech2: OldMechanism, keep_left: bool = False
) -> OldMechanism:
    """Update one mechanism with species and reactions from another.

    Any overlapping species or reactions will be replaced with those of the second
    mechanism.

    :param mech1: First mechanism
    :param mech2: Second mechanism
    :param keep_left: Whether to keep data for `mech1` instead of `mech2`
    :return: Updated mechanism
    """
    mech1, mech2 = (mech2, mech1) if keep_left else (mech1, mech2)

    # Use the rate units of the second mechanism
    mech1, mech2 = common_rate_units_all((mech1, mech2))

    # Get intersection information for the first mechanism
    tmp_col = col_.temp()
    mech1, _ = with_intersection_columns(mech1, mech2, col=tmp_col)

    # Determine combined reactions table
    rxn_df1 = reactions(mech1).filter(~polars.col(tmp_col)).drop(tmp_col)
    rxn_df2 = reactions(mech2)
    rxn_df = polars.concat([rxn_df1, rxn_df2], how="diagonal_relaxed")

    # Determine combined species table
    spc_df1 = species(mech1).filter(~polars.col(tmp_col)).drop(tmp_col)
    spc_df2 = species(mech2)
    spc_df = polars.concat([spc_df1, spc_df2], how="diagonal_relaxed")

    return update_data(mech1, rxn_df=rxn_df, spc_df=spc_df)


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
    spc_df = species(mech1)
    rxn_df = reactions(mech1)

    ncol0 = Species.name
    ncol = col_.prefix(ncol0, col_.temp())
    spc_df = spc_df.with_columns(polars.col(ncol0).alias(ncol))
    spc_df = spec_table.left_update(spc_df, species(mech2), drop_orig=drop_orig)
    rxn_df = reac_table.rename(rxn_df, spc_df[ncol0], spc_df[ncol], drop_orig=drop_orig)
    spc_df = spc_df.drop(ncol)
    rxn_df = reac_table.left_update(rxn_df, reactions(mech2), drop_orig=drop_orig)
    return Mechanism.model_validate(
        {**mech1.model_dump(), "reactions": rxn_df, "species": spc_df}
    )


def with_intersection_columns(
    mech1: OldMechanism,
    mech2: OldMechanism,
    col: str = "intersection",
    stereo: bool = True,
) -> tuple[OldMechanism, OldMechanism]:
    """Add columns to Mechanism pair indicating their intersection.

    :param mech1: First mechanism
    :param mech2: Second mechanism
    :param col: Output column identifying common species and reactions
    :param stereo: Whether to consider stereochemistry
    :return: First and second Mechanisms with intersection columns
    """
    tmp_col = col_.temp()
    mech1 = with_key(mech1, col=tmp_col, stereo=stereo)
    mech2 = with_key(mech2, col=tmp_col, stereo=stereo)

    # Determine species intersection
    spc_df1, spc_df2 = map(species, (mech1, mech2))
    spc_df1, spc_df2 = df_.with_intersection_columns(
        spc_df1, spc_df2, comp_col_=tmp_col, col=col
    )

    # Determine reaction intersection
    rxn_df1, rxn_df2 = map(reactions, (mech1, mech2))
    rxn_df1, rxn_df2 = df_.with_intersection_columns(
        rxn_df1, rxn_df2, comp_col_=tmp_col, col=col
    )

    # Drop temporary columns
    spc_df1, spc_df2 = (df.drop(tmp_col) for df in (spc_df1, spc_df2))
    rxn_df1, rxn_df2 = (df.drop(tmp_col) for df in (rxn_df1, rxn_df2))

    # Return the updated mechanisms
    mech1 = update_data(mech1, rxn_df=rxn_df1, spc_df=spc_df1)
    mech2 = update_data(mech2, rxn_df=rxn_df2, spc_df=spc_df2)
    return mech1, mech2


# parent
def expand_parent_stereo(
    par_mech: OldMechanism, exp_sub_mech: OldMechanism
) -> OldMechanism:
    """Apply stereoexpansion of submechanism to parent mechanism.

    Produces equivalent of parent mechanism, containing distinct
    stereoisomers of submechanism. Expansion is completely naive, with no
    consideration of stereospecificity, and is simply designed to allow merging of
    stereo-expanded submechanism into parent mechanism.

    :param par_mech: Parent mechanism
    :param exp_sub_mech: Stereo-expanded sub-mechanism
    :return: Equivalent parent mechanism, with distinct stereoisomers from
        sub-mechanism
    """
    # 1. Species table
    #   a. Add stereo columns to par_mech species table
    col_dct = col_.to_orig([Species.name, Species.smiles, Species.amchi])
    par_spc_df = species(par_mech)
    par_spc_df = par_spc_df.rename(col_dct)

    #   b. Group by original names and isolate expanded stereoisomers
    sub_spc_df = species(exp_sub_mech)
    sub_spc_df = schema.species_table(sub_spc_df, model_=SpeciesStereo)
    sub_spc_df = sub_spc_df.select(*col_dct.keys(), *col_dct.values())
    sub_spc_df = sub_spc_df.group_by(col_.orig(Species.name)).agg(polars.all())

    #   c. Form species expansion dictionary, to be used for reaction expansion
    exp_dct: dict[str, list[str]] = df_.lookup_dict(
        sub_spc_df, col_.orig(Species.name), Species.name
    )

    #   d. Join on original names, explode, and fill in non-stereoisomer columns
    exp_spc_df = par_spc_df.join(sub_spc_df, how="left", on=col_.orig(Species.name))
    exp_spc_df = exp_spc_df.drop(polars.selectors.ends_with("_right"))
    exp_spc_df = exp_spc_df.explode(*col_dct.keys())
    exp_spc_df = exp_spc_df.with_columns(
        *(polars.col(k).fill_null(polars.col(v)) for k, v in col_dct.items())
    )

    # 2. Reaction table
    #   a. Identify subset of reactions to be expanded
    par_rxn_df = reactions(par_mech)
    has_rate = ReactionRateOld.rate in par_rxn_df
    par_rxn_df = reac_table.with_rates(par_rxn_df)

    par_rxn_df = par_rxn_df.with_columns(
        **col_.from_orig([Reaction.reactants, Reaction.products, ReactionRateOld.rate])
    )
    needs_exp = (
        polars.concat_list(Reaction.reactants, Reaction.products)
        .list.eval(polars.element().is_in(list(exp_dct.keys())))
        .list.any()
    )
    exp_rxn_df = par_rxn_df.filter(needs_exp)
    rem_rxn_df = par_rxn_df.filter(~needs_exp)

    #   b. Expand reactions
    def _expand(rct0s, prd0s, rate0):
        rxn0 = data.reac.from_data(rct0s, prd0s, rate_=rate0)
        rxns = data.reac.expand_lumped_species(rxn0, exp_dct=exp_dct)
        rcts_lst = list(map(data.reac.reactants, rxns))
        prds_lst = list(map(data.reac.products, rxns))
        rates = list(map(dict, map(data.reac.rate_dict, rxns)))
        return rcts_lst, prds_lst, rates

    cols = [Reaction.reactants, Reaction.products, ReactionRateOld.rate]
    dtypes = list(map(polars.List, map(exp_rxn_df.schema.get, cols)))
    exp_rxn_df = df_.map_(exp_rxn_df, cols, cols, _expand, dtype_=dtypes, bar=True)
    exp_rxn_df: polars.DataFrame = exp_rxn_df.explode(cols)
    exp_rxn_df = polars.concat([rem_rxn_df, exp_rxn_df])

    if not has_rate:
        exp_rxn_df = reac_table.without_rates(exp_rxn_df)
        exp_rxn_df = exp_rxn_df.drop(col_.orig(ReactionRateOld.rate))

    return from_data(
        rxn_inp=exp_rxn_df,
        spc_inp=exp_spc_df,
        thermo_temps=thermo_temperatures(par_mech),
        rate_units=rate_units(par_mech),
    )


# building
ReagentValue_ = str | Sequence[str] | None


def enumerate_reactions(
    mech: OldMechanism,
    smarts: str,
    rcts_: Sequence[ReagentValue_] | None = None,
    spc_col_: str | Sequence[str] = Species.name,
    src_mech: OldMechanism | None = None,
    repeat: int = 1,
    drop_self_rxns: bool = True,
) -> OldMechanism:
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
    mech: OldMechanism,
    smarts: str,
    rcts_: Sequence[ReagentValue_] | None = None,
    spc_col_: str | Sequence[str] = Species.name,
    src_mech: OldMechanism | None = None,
) -> OldMechanism:
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
    spc_df = species(mech)
    spc_pool = df_.values(spc_df, spc_col_)
    rcts_ = [spc_pool if r is None else [r] if isinstance(r, str) else r for r in rcts_]

    # Enumerate reactions
    rxn_spc_ids = []
    for rcts in itertools.product(*rcts_):
        rct_spc_ids = spec_table.species_ids(spc_df, rcts, col_=spc_col_, try_fill=True)
        rct_chis, *_ = zip(*rct_spc_ids, strict=True)
        for rxn in automol.reac.enum.from_amchis(smarts, rct_chis):
            _, prd_chis = automol.reac.amchis(rxn)
            prd_spc_ids = spec_table.species_ids(
                spc_df, prd_chis, col_=Species.amchi, try_fill=True
            )
            rxn_spc_ids.append((rct_spc_ids, prd_spc_ids))

    # Form the updated species DataFrame
    spc_ids = list(itertools.chain.from_iterable(r + p for r, p in rxn_spc_ids))
    spc_ids = list(mit.unique_everseen(spc_ids))
    spc_df = spec_table.add_missing_species_by_id(spc_df, spc_ids)
    spc_df = (
        spc_df
        if src_mech is None
        else spec_table.left_update(spc_df, species(src_mech))
    )

    # Form the updated reactions DataFrame
    spc_names = spec_table.species_names_by_id(spc_df, spc_ids)
    name_ = dict(zip(spc_ids, spc_names, strict=True)).get
    rxn_ids = [[list(map(name_, r)) for r in rs] for rs in rxn_spc_ids]
    rxn_ids = list(mit.unique_everseen(rxn_ids))
    rxn_df = reac_table.add_missing_reactions_by_id(reactions(mech), rxn_ids)

    mech = update_data(mech, rxn_df=rxn_df, spc_df=spc_df)
    mech = mech if src_mech is None else left_update(mech, src_mech)
    return drop_duplicate_reactions(mech)


# sorting
def with_sort_data(mech: OldMechanism) -> OldMechanism:
    """Add columns to sort mechanism by species and reactions.

    :param mech: Mechanism
    :return: Mechanism with sort columns
    """
    # Sort species by formula
    spc_df = spec_table.sort_by_formula(species(mech))
    mech = set_species(mech, spc_df)

    # Sort reactions by shape and by reagent names
    idx_col = col_.temp()
    rxn_df = reactions(mech)
    rxn_df = rxn_df.sort(
        polars.col(Reaction.reactants).list.len(),
        polars.col(Reaction.products).list.len(),
        df_.list_to_struct_expression(rxn_df, Reaction.reactants),
        df_.list_to_struct_expression(rxn_df, Reaction.products),
    )
    rxn_df = df_.with_index(rxn_df, idx_col)
    mech = set_reactions(mech, rxn_df)

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
    srt_schema = {idx_col: polars.UInt32, **schema.types([ReactionSorted])}
    srt_df = polars.DataFrame(srt_data, schema=srt_schema)

    # Add sort data to reactions dataframe and sort
    rxn_df = rxn_df.join(srt_df, on=idx_col, how="left")
    rxn_df = rxn_df.drop(idx_col)
    rxn_df = rxn_df.sort(
        ReactionSorted.pes, ReactionSorted.subpes, ReactionSorted.channel
    )
    return set_reactions(mech, rxn_df)


# comparison
def are_equivalent(mech1: OldMechanism, mech2: OldMechanism) -> bool:
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
    same_reactions = reactions(mech1).equals(reactions(mech2))
    same_species = species(mech1).equals(species(mech2))
    return same_reactions and same_species


# read/write
def string(mech: OldMechanism) -> str:
    """Write mechanism to JSON string.

    :param mech: Mechanism
    :return: Mechanism JSON string
    """
    return json.dumps(dict(mech))


def from_string(mech_str: str) -> OldMechanism:
    """Read mechanism from JSON string.

    :param mech_str: Mechanism JSON string
    :return: Mechanism
    """
    mech_dct = json.loads(mech_str)
    return OldMechanism(**mech_dct)


# display
def display(
    mech: OldMechanism,
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
    mech: OldMechanism,
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
    spc_df: polars.DataFrame = species_(mech)

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
    spc_df = df_.map_(spc_df, (Species.amchi, *keys), None, _display_species)


def display_reactions(
    mech: OldMechanism,
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
    spc_df: polars.DataFrame = species(mech)
    rxn_df: polars.DataFrame = reactions(mech)

    chi_dct = df_.lookup_dict(spc_df, Species.name, Species.amchi)
    trans_dcts = {k: df_.lookup_dict(spc_df, Species.name, k) for k in spc_keys}

    rxn_df = df_.map_(
        rxn_df,
        (Reaction.reactants, Reaction.products),
        "eq",
        data.reac.write_chemkin_equation,
    )

    if eqs is not None:
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
    rxn_df = df_.map_(rxn_df, ("eq", *keys), None, _display_reaction)
