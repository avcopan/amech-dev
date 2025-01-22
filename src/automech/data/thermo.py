"""Thermodynamics dataclasses."""

import dataclasses


@dataclasses.dataclass
class Thermo:
    """Species thermodynamic properties parametrization (see cantera.SpeciesThermo).

    Types:
        NASA7   - coeffs: a0, a1, a2, a3, a4, a5, a6
        NASA9   - coeffs: a0, a1, a2, a3, a4, a5, a6, a7, a8
        Shomate - coeffs: A, B, C, D, E, F, G

    :param T0: The minimum temperature [K] at which the parametrization is valid
    :param T_: The maximum temperature [K] at which the parametrization is valid
    :param coeffs: A list of coefficients for the parametrization
    :param type_: The type of parametrization: "NASA8", "NASA9", "Shomate"
    :param P_ref: The reference pressure [Pa] for the parametrization
    """

    T0: float
    T_: float
    coeffs: tuple[float, ...]
    type_: str = "NASA7"
    P_ref: float = 101325  # 1 atm
