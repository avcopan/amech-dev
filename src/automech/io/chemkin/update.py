"""Functions for updating mechanisms from CHEMKIN-formatted files."""

from collections.abc import Sequence

import polars

from ... import _mech, reac_table
from ..._mech import Mechanism
from ...util.io_ import TextInput
from . import read


def thermo(mech: Mechanism, inp_: TextInput | Sequence[TextInput]) -> Mechanism:
    """Update thermochemical data in mechanism.

    :param mech: Mechanism
    :param inp_: ChemKin file(s) or string(s)
    :return: Mechanism
    """
    inp_ = [inp_] if isinstance(inp_, TextInput) else inp_

    spc_df = _mech.species(mech)
    for inp in inp_:
        spc_df = read.thermo(inp, spc_df=spc_df)
    return _mech.set_species(mech, spc_df)


def rates(mech: Mechanism, inp_: TextInput | Sequence[TextInput]) -> Mechanism:
    """Update rate data in mechanism.

    :param mech: Mechanism
    :param inp_: ChemKin file(s) or string(s)
    :return: Mechanism
    """
    inp_ = [inp_] if isinstance(inp_, TextInput) else inp_
    units = _mech.rate_units(mech)
    spc_df = _mech.species(mech)
    rxn_dfs = []
    for inp in inp_:
        rxn_df, err = read.reactions(inp, units=units, spc_df=spc_df)
        rxn_dfs.append(rxn_df)
        assert err.is_empty(), f"\ninp = {inp}\nerr = {err}"

    rxn_df0 = _mech.reactions(mech)
    rxn_df = reac_table.left_update_rates(rxn_df0, polars.concat(rxn_dfs))
    return _mech.set_reactions(mech, rxn_df)
