"""DataFrame schema."""

from collections.abc import Sequence
from typing import Annotated

import automol
import polars
import pydantic
from pydantic_core import core_schema

from ..util import df_, model_
from ..util.model_ import Model


class Species(Model):
    """Core species table."""

    name: str
    smiles: str
    amchi: str
    spin: int
    charge: int
    formula: polars.Struct


SpeciesDataFrame_ = Annotated[
    pydantic.SkipValidation[polars.DataFrame],
    pydantic.BeforeValidator(polars.DataFrame),
    pydantic.AfterValidator(Species.validate),
    pydantic.PlainSerializer(lambda x: polars.DataFrame(x).to_dict(as_series=False)),
    pydantic.GetPydanticSchema(
        lambda _, handler: core_schema.with_default_schema(handler(dict[str, list]))
    ),
]


def bootstrap(
    data: dict[str, Sequence[object]],
    name_dct: dict[str, str] | None = None,
    key: str = Species.amchi,
) -> polars.DataFrame:
    """Bootstrap species DataFrame from minimal data.

    :param df: DataFrame
    :param name_dct: Names by key
    :param key: Key for filling from dictionaries, 'smiles' or 'amchi'.
    :return: DataFrame
    """
    # 0. Make dataframe from given data
    df = polars.DataFrame(data, strict=False)

    # 1. Get spin from multiplicity if given
    if Species.spin not in df and "mult" in df:
        df = df.with_columns((polars.col("mult") - 1).cast(int).alias(Species.spin))

    # 3. Add missing spin column
    df = model_.add_missing_column_(Species, df, Species.spin)

    # 4. Sanitize SMILES with spin tags
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

    # 5. Populate AMChIs from SMILES or InChI
    if Species.amchi not in df and Species.smiles in df:
        df = df_.map_(df, Species.smiles, Species.amchi, automol.smiles.amchi, bar=True)
    if Species.amchi not in df and "inchi" in df:
        df = df_.map_(df, "inchi", Species.amchi, automol.inchi.amchi, bar=True)

    # 6. Populate missing columns from AMChI
    schema = model_.schema(Species)
    populators = {
        Species.name: automol.amchi.chemkin_name,
        Species.spin: automol.amchi.guess_spin,
        Species.charge: (lambda _: 0),
        Species.formula: automol.amchi.formula,
    }
    for col, pop_ in populators.items():
        dtype = schema.get(col)
        expr = polars.col(Species.amchi).map_elements(pop_, return_dtype=dtype)
        df = df.with_columns(expr.alias(col))

    # 7. Replace names if given
    if name_dct is not None:
        assert key in (Species.smiles, Species.amchi), f"Invalid key: {key}"
        if key == Species.smiles:
            name_dct = {automol.smiles.amchi(k): v for k, v in name_dct.items()}
        orig = polars.col(Species.name)
        expr = polars.col(Species.amchi).replace_strict(name_dct, default=orig)
        df = df.with_columns(expr.alias(Species.name))

    return model_.validate(Species, df)
