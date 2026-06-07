"""
Gamut Area Index
================

Define the *Gamut Area Index* (GAI) computation objects.

-   :class:`colour.quality.ColourQuality_Specification_GAI`
-   :func:`colour.gamut_area_index`

References
----------
-   :cite:`Rea2008` : Rea, M. S., & Freyssinier-Nova, J. P. (2008). Color
    rendering: A tale of two metrics. Color Research & Application, 33(3),
    192-202. doi:10.1002/col.20399
-   :cite:`Freyssinier2010` : Freyssinier, J. P., & Rea, M. (2010). A
    two-metric proposal to specify the color-rendering properties of light
    sources for retail lighting. Proceedings of SPIE, 7784, 77840V.
    doi:10.1117/12.863063
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np

from colour.colorimetry import (
    MSDS_CMFS,
    SDS_ILLUMINANTS,
    SPECTRAL_SHAPE_DEFAULT,
    MultiSpectralDistributions,
    SpectralDistribution,
    reshape_msds,
    reshape_sd,
    sd_to_XYZ,
)

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, Dict, Literal, NDArrayFloat, Tuple

from colour.models import XYZ_to_xy, xy_to_Luv_uv
from colour.quality.datasets.tcs import INDEXES_TO_NAMES_TCS, SDS_TCS
from colour.utilities import as_float_array, as_float_scalar

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "DataColorimetry_TCS_GAI",
    "ColourQuality_Specification_GAI",
    "gamut_area_index",
    "tcs_colorimetry_data",
]

_TCS8_NAMES: tuple = tuple(
    value for key, value in sorted(INDEXES_TO_NAMES_TCS["CIE 1995"].items()) if key <= 8
)
"""Names of the first eight *CIE 1995* *test colour samples*."""


@dataclass
class DataColorimetry_TCS_GAI:
    """
    Store colorimetric data for *test colour samples* used in *Gamut Area
    Index* calculations.

    Attributes
    ----------
    name
        Identifier for the test colour sample.
    XYZ
        *CIE XYZ* tristimulus values of the test colour sample.
    uv
        *CIE 1976 UCS* :math:`u'v'` chromaticity coordinates of the test
        colour sample.
    """

    name: str
    XYZ: NDArrayFloat
    uv: NDArrayFloat


@dataclass
class ColourQuality_Specification_GAI:
    """
    Define the *Gamut Area Index* (GAI) colour quality specification.

    Parameters
    ----------
    name
        Name of the test spectral distribution.
    Q_g
        *Gamut Area Index* (GAI) :math:`Q_g` value.
    gamut_area
        Gamut area of the test spectral distribution in the *CIE 1976 UCS*
        :math:`u'v'` diagram.
    reference_gamut_area
        Gamut area of the equal-energy reference spectral distribution in the
        *CIE 1976 UCS* :math:`u'v'` diagram.
    colorimetry_data
        Colorimetry data for the test and equal-energy reference illuminant
        computations.

    References
    ----------
    :cite:`Rea2008`, :cite:`Freyssinier2010`
    """

    name: str
    Q_g: float
    gamut_area: float
    reference_gamut_area: float
    colorimetry_data: Tuple[
        Tuple[DataColorimetry_TCS_GAI, ...], Tuple[DataColorimetry_TCS_GAI, ...]
    ]


@typing.overload
def gamut_area_index(
    sd_test: SpectralDistribution,
    additional_data: Literal[False] = False,
) -> float: ...


@typing.overload
def gamut_area_index(
    sd_test: SpectralDistribution,
    *,
    additional_data: Literal[True],
) -> ColourQuality_Specification_GAI: ...


@typing.overload
def gamut_area_index(
    sd_test: SpectralDistribution,
    additional_data: Literal[True],
) -> ColourQuality_Specification_GAI: ...


def gamut_area_index(
    sd_test: SpectralDistribution,
    additional_data: bool = False,
) -> float | ColourQuality_Specification_GAI:
    """
    Compute the *Gamut Area Index* (GAI) of the specified spectral
    distribution.

    The *Gamut Area Index* is computed from the area of the polygon formed by
    the first eight *CIE 1995* *test colour samples* in the *CIE 1976 UCS*
    :math:`u'v'` diagram, scaled so that the equal-energy illuminant has a GAI
    value of 100.

    .. math::

        Q_g = 100 \\frac{G_{test}}{G_E}

    Notes
    -----
    The implementation follows the published GAI procedure: the first eight
    *CIE 1995* *test colour samples* are illuminated by the test source and by
    the equal-energy reference source, converted to *CIE 1976 UCS*
    :math:`u'v'` chromaticity coordinates, and compared by the ratio of their
    polygon areas. Spectral alignment, tristimulus integration, and use of the
    *CIE 1931 2 Degree Standard Observer* follow the existing
    :mod:`colour.quality.cri` colourimetry pipeline.

    Parameters
    ----------
    sd_test
        Test spectral distribution.
    additional_data
        Whether to output additional data.

    Returns
    -------
    :class:`float` or :class:`colour.quality.ColourQuality_Specification_GAI`
        *Gamut Area Index* (GAI).

    References
    ----------
    :cite:`Rea2008`, :cite:`Freyssinier2010`

    Examples
    --------
    >>> from colour import SDS_ILLUMINANTS
    >>> sd = SDS_ILLUMINANTS["D65"]
    >>> gamut_area_index(sd)  # doctest: +ELLIPSIS
    np.float64(97.5534517...)
    """

    cmfs = reshape_msds(
        MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
        SPECTRAL_SHAPE_DEFAULT,
        copy=False,
    )

    shape = cmfs.shape
    sd_test = reshape_sd(sd_test, shape, copy=False)
    sd_reference = reshape_sd(SDS_ILLUMINANTS["E"], shape, copy=False)
    tcs_sds = {
        name: reshape_sd(SDS_TCS["CIE 1995"][name], shape, copy=False)
        for name in _TCS8_NAMES
    }

    test_tcs_colorimetry_data = tcs_colorimetry_data(sd_test, tcs_sds, cmfs)
    reference_tcs_colorimetry_data = tcs_colorimetry_data(sd_reference, tcs_sds, cmfs)

    gamut_area = _gamut_area([sample.uv for sample in test_tcs_colorimetry_data])
    reference_gamut_area = _gamut_area(
        [sample.uv for sample in reference_tcs_colorimetry_data]
    )

    Q_g = 100 * gamut_area / reference_gamut_area

    if additional_data:
        return ColourQuality_Specification_GAI(
            sd_test.name,
            Q_g,
            gamut_area,
            reference_gamut_area,
            (test_tcs_colorimetry_data, reference_tcs_colorimetry_data),
        )

    return Q_g


def tcs_colorimetry_data(
    sd_test: SpectralDistribution,
    sds_tcs: Dict[str, SpectralDistribution],
    cmfs: MultiSpectralDistributions,
) -> Tuple[DataColorimetry_TCS_GAI, ...]:
    """
    Compute the *test colour samples* colorimetry data used in *Gamut Area
    Index* calculations.

    The samples are evaluated directly under the specified illuminant because
    GAI compares the test-source gamut area against the equal-energy reference
    gamut area in :math:`u'v'` space.

    Parameters
    ----------
    sd_test
        Test spectral distribution.
    sds_tcs
        *Test colour samples* spectral reflectance distributions.
    cmfs
        Standard observer colour matching functions.

    Returns
    -------
    :class:`tuple`
        *Test colour samples* colorimetry data.
    """

    tcs_data = []
    for name in _TCS8_NAMES:
        if name not in sds_tcs:
            continue

        sd_tcs = sds_tcs[name]

        XYZ_tcs = sd_to_XYZ(sd_tcs, cmfs, sd_test)
        uv_tcs = xy_to_Luv_uv(XYZ_to_xy(XYZ_tcs))

        tcs_data.append(DataColorimetry_TCS_GAI(sd_tcs.name, XYZ_tcs, uv_tcs))

    return tuple(tcs_data)


def _gamut_area(uv: ArrayLike) -> float:
    """
    Compute the area of the :math:`u'v'` polygon used by the *Gamut Area
    Index*.

    .. math::

        G = \\frac{1}{2}\\left|\\sum_i
        \\left(u'_i v'_{i+1} - v'_i u'_{i+1}\\right)\\right|
    """

    uv = as_float_array(uv)
    uv_next = np.roll(uv, -1, axis=0)

    # Planar polygon area in the specified CIE 1976 UCS u'v' diagram.
    return abs(
        as_float_scalar(
            np.sum((uv[..., 0] * uv_next[..., 1]) - (uv[..., 1] * uv_next[..., 0])) / 2
        )
    )
