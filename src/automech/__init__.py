"""Data processing at the level of whole mechanisms."""

from . import io, net, reac_table, spec_table, util
from ._mech import (
    Mechanism,
    are_equivalent,
    combine_all,
    difference,
    display,
    display_reactions,
    display_species,
    drop_duplicate_reactions,
    drop_self_reactions,
    enumerate_reactions,
    expand_parent_stereo,
    expand_stereo,
    from_network,
    from_smiles,
    from_string,
    intersection,
    neighborhood,
    network,
    reaction_count,
    rename,
    rename_dict,
    species_count,
    species_names,
    string,
    update,
    with_intersection_columns,
    with_key,
    with_sort_data,
    with_species,
    without_unused_species,
)

__all__ = [
    # types
    "Mechanism",
    # functions
    "from_network",
    "from_smiles",
    # properties
    "species_count",
    "reaction_count",
    "species_names",
    "rename_dict",
    "network",
    # add/remove reactions
    "drop_duplicate_reactions",
    "drop_self_reactions",
    # transformations
    "rename",
    "neighborhood",
    "with_species",
    "without_unused_species",
    "with_key",
    "expand_stereo",
    # binary operations
    "combine_all",
    "intersection",
    "difference",
    "update",
    "with_intersection_columns",
    # parent
    "expand_parent_stereo",
    # building
    "enumerate_reactions",
    # sorting
    "with_sort_data",
    # comparisons
    "are_equivalent",
    # read/write,
    "string",
    "from_string",
    # display
    "display",
    "display_species",
    "display_reactions",
    # modules
    "io",
    "util",
    "net",
    "reac_table",
    "spec_table",
]
