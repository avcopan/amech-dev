"""Reaction dataclasses."""

import dataclasses
import itertools
import math
import re
from collections.abc import Sequence

import more_itertools as mit
import pyparsing as pp

from . import rate as rt_
from .rate import Rate, RateType


@dataclasses.dataclass
class Reaction:
    """A reaction.

    :param reactants: Names for the reactants
    :param products: Names for the products
    :param rate: The reaction rate
    :param colliders: Colliders along with their efficiencies, e.g. {"M": 1, "Ar": 1.2}
    """

    reactants: tuple[str, ...]
    products: tuple[str, ...]
    rate: Rate | None = None
    colliders: dict[str, float] | None = None

    def __post_init__(self):
        """Initialize attributes."""
        self.reactants = tuple(map(str, self.reactants))
        self.products = tuple(map(str, self.products))

        if not self.colliders or all(v is None for v in self.colliders.values()):
            self.colliders = None

        if rt_.needs_collider(self.rate) and self.colliders is None:
            self.colliders = {"M": 1.0}

        if self.colliders is not None:
            self.colliders = {k: v for k, v in self.colliders.items() if v is not None}


# constructors
def from_data(
    rcts: Sequence[str],
    prds: Sequence[str],
    rate_: Rate | None = None,
    coll_dct: dict[str, float] | None = None,
) -> Reaction:
    """Construct a reaction object from data.

    :param rcts: Names for the reactants
    :param prds: Names for the products
    :param rate_: The reaction rate
    :param coll_dct: Colliders along with their efficiencies, e.g. {"M": 1, "Ar": 1.2}
    :return: The reaction object
    """
    rate_ = rt_.from_data(**rate_) if isinstance(rate_, dict) else rate_
    return Reaction(reactants=rcts, products=prds, rate=rate_, colliders=coll_dct)


def from_equation(
    eq: str, rate_: Rate | None = None, coll_dct: dict[str, float] | None = None
) -> Reaction:
    """Construct a reaction object from a CHEMKIN equation.

    :param eq: The equation
    :param rate_: The reaction rate, defaults to None
    :param coll_dct: Colliders along with their efficiencies, e.g. {"M": 1, "Ar": 1.2}
    :return: The reaction object
    """
    rcts, prds, coll, *_ = read_chemkin_equation(eq, bare_coll=True)
    if coll is not None:
        coll_dct = {coll: 1.0} if coll_dct is None else {coll: 1.0, **coll_dct}

    return from_data(rcts=rcts, prds=prds, rate_=rate_, coll_dct=coll_dct)


# getters
def reactants(rxn: Reaction) -> tuple[str, ...]:
    """Get the list of reactants.

    :param rxn: A reaction object
    :return: The CHEMKIN names of the reactants
    """
    return rxn.reactants


def products(rxn: Reaction) -> tuple[str, ...]:
    """Get the list of products.

    :param rxn: A reaction object
    :return: The CHEMKIN names of the products
    """
    return rxn.products


def rate(rxn: Reaction) -> Rate:
    """Get the rate constant.

    :param rxn: A reaction object
    :return: The rate object
    """
    return rxn.rate


def rate_dict(rxn: Reaction) -> dict[str, object]:
    """Get the rate constant.

    :param rxn: A reaction object
    :return: The rate object
    """
    return rxn.rate.model_dump()


def colliders(rxn: Reaction, aux_only: bool = False) -> dict[str, float] | None:
    """Get the collider, if there is one.

    :param rxn: A reaction object
    :param aux_only: Only return the auxiliary colliders?
    :return: A dictionary mapping collider names onto efficiencies
    """
    coll_dct = rxn.colliders

    if coll_dct is None:
        return None

    if aux_only:
        coll0 = primary_collider(rxn)
        coll_dct = {k: v for k, v in coll_dct.items() if k != coll0}

    return coll_dct


# setters
def set_reactants(rxn: Reaction, rcts: tuple[str, ...]) -> Reaction:
    """Set the list of reactants.

    :param rxn: A reaction object
    :param rcts: The new CHEMKIN names of the reactants
    :return: The new reaction object
    """
    return from_data(
        rcts=rcts, prds=products(rxn), rate_=rate(rxn), coll_dct=colliders(rxn)
    )


def set_products(rxn: Reaction, prds: tuple[str, ...]) -> Reaction:
    """Set the list of products.

    :param rxn: A reaction object
    :param prds: The CHEMKIN names of the products
    :return: The new reaction object
    """
    return from_data(
        rcts=reactants(rxn), prds=prds, rate_=rate(rxn), coll_dct=colliders(rxn)
    )


def set_rate(rxn: Reaction, rate_: Rate) -> Reaction:
    """Set the rate constant.

    :param rxn: A reaction object
    :param rate_: The rate object
    :return: The new reaction object
    """
    return from_data(
        rcts=reactants(rxn), prds=products(rxn), rate_=rate_, coll_dct=colliders(rxn)
    )


def set_colliders(
    rxn: Reaction, coll_dct: dict[str, float] | None, aux_only: bool = False
) -> Reaction:
    """Set the collider, if there is one.

    :param rxn: A reaction object
    :param coll_dct: A dictionary mapping collider names onto efficiencies
    :param aux_only: Only set the auxiliary colliders?
    :return: The new reaction object
    """
    coll0 = primary_collider(rxn)
    if aux_only and coll0 is not None:
        coll_dct = {coll0: 1, **coll_dct}

    return from_data(
        rcts=reactants(rxn), prds=products(rxn), rate_=rate(rxn), coll_dct=coll_dct
    )


# properties
def species(rxn: Reaction) -> tuple[str, ...]:
    """Get the species that are involved in the reaction.

    :param rxn: A reaction object
    :return: The list of species
    """
    rcts = reactants(rxn)
    prds = products(rxn)
    return tuple(mit.unique_everseen(rcts + prds))


def rate_type(rxn: Reaction) -> RateType:
    """Get the rate type for a reaction object.

    :param rxn: A reaction object
    :return: The rate type
    """
    return rt_.type_(rate(rxn))


def primary_collider(rxn: Reaction) -> str | None:
    """Get the primary collider, if there is one.

    This is the collider that would appear in the chemical equation.

    :param rxn: A reaction object
    :return: The primary collider
    """
    coll_dct = rxn.colliders

    if coll_dct is None:
        return None

    if "M" in coll_dct:
        eff = coll_dct["M"]
        assert eff == 1.0, f"Invalid efficiency {eff} for generic collider 'M'"
        return "M"

    coll_name = next((k for k, v in sorted(coll_dct.items()) if v == 1.0), None)
    return coll_name


def is_falloff(rxn: Reaction) -> bool:
    """Whether this is a fallof reaction.

    :param rxn: A reaction object
    :return: `True` if it is, `False` if it isn't
    """
    return rt_.is_falloff(rate(rxn))


def equation(rxn: Reaction, sort_reag: bool = False, sort_dir: bool = False) -> str:
    """Get the CHEMKIN equation of a reaction (excludes collider term).

    :param rxn: A reaction object
    :param sort_reag: Sort the reagents on each side of the reaction?
    :param sort_dir: Sort the direction of the reaction?
    :return: The reaction CHEMKIN equation
    """
    return write_chemkin_equation(
        reactants(rxn), products(rxn), sort_reag=sort_reag, sort_dir=sort_dir
    )


def chemkin_collider(rxn: Reaction) -> str | None:
    """Get the CHEMKIN collider name for a reaction.

    :param rxn: A reaction object
    :return: The CHEMKIN collider name
    """
    coll = primary_collider(rxn)
    return f"(+{coll})" if is_falloff(rxn) else coll


def chemkin_reagents(
    rxn: Reaction, tuple_coll: bool = False, sort: bool = False
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Get the CHEMKIN reactants, products, and collider names for a reaction.

    :param rxn: A reaction object
    :param tuple_coll: Whether to return the collider as a tuple, defaults to False
    :param sort: Sort the reactants and products alphabetically?
    :return: The CHEMKIN names of the reactants, products, and collider(s)
    """
    rcts = reactants(rxn)
    prds = products(rxn)
    coll = chemkin_collider(rxn)
    if sort:
        rcts = tuple(sorted(rcts))
        prds = tuple(sorted(prds))
    return rcts, prds, (coll,) if tuple_coll else coll


def chemkin_equation(
    rxn: Reaction, sort_reag: bool = False, sort_dir: bool = False
) -> str:
    """Get the CHEMKIN equation of a reaction (includes collider term).

    :param rxn: A reaction object
    :param sort_reag: Sort the reagents on each side of the reaction?
    :param sort_dir: Sort the direction of the reaction?
    :return: The reaction CHEMKIN equation
    """
    is_rev = rt_.is_reversible(rate(rxn))
    rcts, prds, coll = chemkin_reagents(rxn, sort=sort_reag)
    arrow = "=" if is_rev else "=>"
    return write_chemkin_equation(
        rcts,
        prds,
        coll=coll,
        arrow=arrow,
        sort_reag=sort_reag,
        sort_dir=sort_dir,
    )


def chemkin_duplicate_key(rxn: Reaction) -> str:
    """Return a key for detecting duplicates in a CHEMKIN file.

    :param rxn: A reaction object
    :return: The reaction CHEMKIN equation
    """
    is_rev = rt_.is_reversible(rate(rxn))
    return chemkin_equation(rxn, sort_reag=True, sort_dir=is_rev)


# transformations
def convert_energy_units(rxn: Reaction, unit0: str, unit: str) -> Reaction:
    """Convert the energy units for a reaction rate.

    :param rxn: A reaction object
    :param unit0: The current energy unit
    :param unit: The desired energy unit
    :return: The reaction object, with new rate units
    """
    return set_rate(rxn, rt_.convert_energy_units(rate(rxn), unit0, unit))


def expand_lumped_species(
    rxn: Reaction, exp_dct: dict[str, Sequence[str]]
) -> list[Reaction]:
    """Naively expand lumped species in a reaction, reproducing its effective rate.

    General formula for unlumped rate coefficient, assuming an even ratio:

        unlumped rate coefficient
        = lumped rate coefficient x
            nexp ^ stoich / multiset(nexp, stoich) for each lumped reactant
            1             / multiset(nexp, stoich) for each lumped product

    where nexp is the number of components in the lump and stoich is its stoichiometry
    in the reaction.

    :param rxn: A reaction object
    :param exp_dct: A mapping of lumped species names onto unlumped species name sets
    :return: The expanded sequence of unlumped reactions
    """
    rxn0 = rxn

    def _expand(rgt0: str, prod: bool) -> tuple[list[dict[int, str]], float]:
        """Determine the expansion of reactants or products, along with the factor for
        scaling the rate.

        :param rgt: The name of the reagent
        :param prod: Is this reagent a product?
        :return: A list of dictionaries mapping reagent list indices onto new names,
            and a factor for scaling the rate constant
        """
        rgt0s = products(rxn0) if prod else reactants(rxn0)
        # Build the expansion
        stoich = rgt0s.count(rgt0)
        spc_exp = exp_dct.get(rgt0)
        rgt_combs = list(itertools.combinations_with_replacement(spc_exp, stoich))
        rgt_idxs = [i for i, r in enumerate(rgt0s) if r == rgt0]
        rgt_exp_dcts = [dict(zip(rgt_idxs, c, strict=True)) for c in rgt_combs]
        # Determine the factor
        factor = 1 if prod else len(spc_exp) ** stoich
        factor /= len(rgt_combs)
        return rgt_exp_dcts, factor

    rct0s = reactants(rxn0)
    prd0s = products(rxn0)
    coll_dct = colliders(rxn0)

    rexps = [_expand(s, prod=False) for s in set(rct0s) if s in exp_dct]
    pexps = [_expand(s, prod=True) for s in set(prd0s) if s in exp_dct]

    rexp_dcts_lst, rfactors = zip(*rexps, strict=True) if rexps else ((), ())
    pexp_dcts_lst, pfactors = zip(*pexps, strict=True) if pexps else ((), ())

    # Scale the rate by the calculated factor
    factor = math.prod(rfactors + pfactors)
    rate_ = rate(rxn0) * factor

    # Expand all possible combinations of un-lumped species
    rexp_dcts = [
        {k: v for d in ds for k, v in d.items()}
        for ds in itertools.product(*rexp_dcts_lst)
    ]
    pexp_dcts = [
        {k: v for d in ds for k, v in d.items()}
        for ds in itertools.product(*pexp_dcts_lst)
    ]
    rxns = []
    for rexp_dct, pexp_dct in itertools.product(rexp_dcts, pexp_dcts):
        rcts = [rexp_dct.get(i) if i in rexp_dct else s for i, s in enumerate(rct0s)]
        prds = [pexp_dct.get(i) if i in pexp_dct else s for i, s in enumerate(prd0s)]
        rxns.append(from_data(rcts=rcts, prds=prds, rate_=rate_, coll_dct=coll_dct))

    return rxns


# I/O
def chemkin_string(rxn: Reaction, eq_width: int = 55, dup: bool = False) -> str:
    """Write a Reaction object to a CHEMKIN string.

    :param rxn: A reaction object
    :param eq_width: The column width for the reaction equation
    :param dup: Add the "DUPLICATE" keyword to this reaction?
    :return: The CHEMKIN reaction string
    """
    eq = chemkin_equation(rxn)
    rate_ = rate(rxn)
    rate_str = rt_.chemkin_string(rate_, eq_width=eq_width)
    rxn_str = f"{eq:<{eq_width}} {rate_str}"

    coll_dct = colliders(rxn, aux_only=True)
    if coll_dct is not None:
        coll_str = " ".join(f"{k}/{v:.4}/" for k, v in coll_dct.items())
        rxn_str += f"\n    {coll_str}"

    if dup:
        rxn_str += "\n    DUPLICATE"

    return f"{rxn_str}"


def from_chemkin_string(rxn_str: str) -> Reaction:
    """Get a Reaction object from a CHEMKIN string.

    :param rxn_str: CHEMKIN reaction data
    :return: The reaction object
    """
    rcts, prds, coll, arrow = read_chemkin_equation(rxn_str, bare_coll=True)
    is_rev = arrow in ("=", "<=>")
    rate_, coll_dct = rt_.from_chemkin_string(rxn_str, is_rev=is_rev, coll=coll)
    return from_data(rcts=rcts, prds=prds, rate_=rate_, coll_dct=coll_dct)


# Chemkin equation helpers
def standardize_chemkin_equation(
    eq: str,
    sort_reag: bool = False,
    sort_dir: bool = False,
    trans_dct: dict[str, str] | None = None,
) -> str:
    """Standardize the format of a CHEMKIN equation for string comparison.

    :param eq: The reaction CHEMKIN equation
    :param sort_reag: Sort the reagents on each side of the reaction?
    :param sort_dir: Sort the direction of the reaction?
    :param trans_dct: Optionally, translate the species names using a dictionary
    :return: The reaction CHEMKIN equation in standard format
    """
    return write_chemkin_equation(
        *read_chemkin_equation(eq),
        sort_reag=sort_reag,
        sort_dir=sort_dir,
        trans_dct=trans_dct,
    )


def write_chemkin_equation(
    rcts: tuple[str],
    prds: tuple[str],
    coll: str | None = None,
    arrow: str = "=",
    trans_dct: dict[str, str] | None = None,
    sort_reag: bool = False,
    sort_dir: bool = False,
) -> str:
    """Write the CHEMKIN equation of a reaction to a string.

    :param rcts: The reactant names
    :param prds: The product names
    :param coll: The collider
    :param trans_dct: Optionally, translate the species names using a dictionary
    :param sort_reag: Sort the reagents on each side of the reaction?
    :param sort_dir: Sort the direction of the reaction?
    :return: The reaction CHEMKIN equation
    """

    def trans_(name):
        return name if trans_dct is None else trans_dct.get(name)

    rcts_ = list(map(trans_, rcts))
    prds_ = list(map(trans_, prds))

    if sort_reag:
        rcts_ = sorted(rcts_)
        prds_ = sorted(prds_)

    if sort_dir:
        rcts_, prds_ = sorted([rcts_, prds_])

    assert all(
        isinstance(n, str) for n in rcts_ + prds_
    ), f"Some species in {rcts}={prds} have no translation:\n{trans_dct}"

    rcts_str = " + ".join(rcts_)
    prds_str = " + ".join(prds_)

    if coll is not None:
        coll = (
            coll if isinstance(coll, str) else coll[0]
        )  # For mechanalyzer compatibility
        sep = " " if "+" in coll else " + "
        rcts_str = sep.join([rcts_str, coll])
        prds_str = sep.join([prds_str, coll])

    return f" {arrow} ".join([rcts_str, prds_str])


def read_chemkin_equation(
    rxn_str: str,
    trans_dct: dict[str, str] | None = None,
    bare_coll: bool = False,
    tuple_coll: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Parse the CHEMKIN equation of a reaction from a string.

    :param rxn_str: CHEMKIN reaction data
    :param trans_dct: Optionally, translate the species names using a dictionary
    :param bare_coll: Return a bare collider, without parentheses?
    :param tuple_coll: Use tuple colliders, for mechanalyzer compatibility? (temporary)
    :return: The reactants and products, along with the colliders
    """

    def trans_(name):
        return name if trans_dct is None else trans_dct.get(name)

    # 0. Find the equation
    rxn_eq = pp.Suppress(...) + pp.Combine(
        pp.OneOrMore(pp.Word(pp.printables), stop_on=rt_.ARRH_PARAMS), adjacent=False
    )("eq")
    res = rxn_eq.parseString(rxn_str)
    eq = res.get("eq")

    # 1. Find the arrow and split
    eq_expr = (
        pp.SkipTo(rt_.ARROW)("side1")
        + rt_.ARROW("arrow")
        + pp.SkipTo(pp.StringEnd())("side2")
    )
    res = eq_expr.parseString(eq)
    arrow = res.get("arrow")

    # 2. For each side, find the collider and the reagents
    rcts, falloff = read_chemkin_equation_side(res.get("side1"))
    prds, falloff_ = read_chemkin_equation_side(res.get("side2"))
    assert not (falloff is None) ^ (
        falloff_ is None
    ), f"Inconsistent equation: {rxn_str}"

    rcts, prds, coll = extract_collider(rcts, prds, falloff)
    rcts = tuple(map(trans_, rcts))
    prds = tuple(map(trans_, prds))

    # If requested, remove (+ ) for falloff reactions and return only the bare collider
    if bare_coll:
        if coll is not None and coll.startswith("("):
            coll = coll[1:-1].lstrip(" +")

    if tuple_coll:
        coll = (coll,)

    return (rcts, prds, coll, arrow)


# Extra helpers
def read_chemkin_equation_side(side: str) -> tuple[list[str], str | None]:
    """Read one side of a CHEMKIN equation.

    :param side: One side of a CHEMKIN equation
    :return: The reagent names and the collider, if any
    """
    # Use pyparsing to split off the falloff value, if present
    end = rt_.FALLOFF | pp.StringEnd()
    side_expr = (
        pp.SkipTo(end)("side") + pp.Optional(rt_.FALLOFF)("falloff") + pp.StringEnd()
    )
    res = side_expr.parseString(side)
    side = res.get("side")
    coll = res.get("falloff")
    # Use re to split the list of species names
    names = re.split(r"\+(?!\+|$)", side)
    return names, coll


def extract_collider(
    rcts: tuple[str, ...],
    prds: tuple[str, ...],
    falloff: str | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], str | None]:
    """Extract a collider from a list of reactants and products.

    :param rcts: The reactant names
    :param prds: The product names
    :param falloff: A falloff string, if present
    :return: The reactants and products (without collider), and the collider
    """
    colliders = ("M", "He", "Ne", "Ar")

    coll = falloff
    if rcts[-1] == prds[-1] and rcts[-1] in colliders:
        assert falloff is None, f"Collider inconsistency: {falloff} {rcts} {prds}"
        coll = rcts[-1]
        rcts = rcts[:-1]
        prds = prds[:-1]

    return tuple(rcts), tuple(prds), coll
