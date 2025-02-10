"""Type utilities."""

from typing import Annotated

import polars
import pydantic
from pydantic_core import core_schema

DataFrame_ = Annotated[
    pydantic.SkipValidation[polars.DataFrame],
    pydantic.BeforeValidator(polars.DataFrame),
    pydantic.PlainSerializer(lambda x: polars.DataFrame(x).to_dict(as_series=False)),
    pydantic.GetPydanticSchema(
        lambda _, handler: core_schema.with_default_schema(handler(dict[str, list]))
    ),
]
