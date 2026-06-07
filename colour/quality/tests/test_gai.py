"""Define the unit tests for the :mod:`colour.quality.gai` module."""

from __future__ import annotations

import numpy as np

from colour.colorimetry import SDS_ILLUMINANTS
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.quality import ColourQuality_Specification_GAI, gamut_area_index
from colour.quality.gai import DataColorimetry_TCS_GAI
from colour.utilities import domain_range_scale

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestGamutAreaIndex",
]


class TestGamutAreaIndex:
    """
    Define :func:`colour.quality.gai.gamut_area_index` definition unit tests
    methods.

    Notes
    -----
    -   The equal-energy illuminant value is definitional: the GAI reference
        gamut area is scaled to 100. The remaining illuminant values are
        implementation regression checks computed from the same documented
        GAI procedure.
    """

    def test_gamut_area_index(self) -> None:
        """Test :func:`colour.quality.gai.gamut_area_index` definition."""

        np.testing.assert_allclose(
            gamut_area_index(SDS_ILLUMINANTS["E"], additional_data=False),
            100.000000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            gamut_area_index(SDS_ILLUMINANTS["D50"], additional_data=False),
            87.943634683785400,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            gamut_area_index(SDS_ILLUMINANTS["D65"], additional_data=False),
            97.553451752639190,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            gamut_area_index(SDS_ILLUMINANTS["FL2"], additional_data=False),
            64.272577899775810,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            gamut_area_index(SDS_ILLUMINANTS["A"], additional_data=False),
            53.298498356551520,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        specification = gamut_area_index(SDS_ILLUMINANTS["D65"], additional_data=True)

        assert isinstance(specification, ColourQuality_Specification_GAI)
        assert specification.name == "D65"
        np.testing.assert_allclose(
            specification.Q_g,
            97.553451752639190,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            specification.gamut_area,
            0.007188856335126,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            specification.colorimetry_data[0][0].XYZ,
            np.array([32.99336849, 29.78373170, 24.51292658]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            specification.reference_gamut_area,
            0.007369146048624,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        assert len(specification.colorimetry_data) == 2
        assert len(specification.colorimetry_data[0]) == 8
        assert len(specification.colorimetry_data[1]) == 8
        assert all(
            isinstance(data, DataColorimetry_TCS_GAI)
            for data in specification.colorimetry_data[0]
        )

        np.testing.assert_allclose(
            specification.colorimetry_data[0][0].uv,
            np.array([0.23852577, 0.48447377]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_gamut_area_index(self) -> None:
        """
        Test :func:`colour.quality.gai.gamut_area_index` definition domain
        range scale support.
        """

        for scale in ("reference", "1", "100"):
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    gamut_area_index(SDS_ILLUMINANTS["D65"], additional_data=False),
                    97.553451752639190,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
