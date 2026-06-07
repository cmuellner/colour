"""
Demonstrate *Gamut Area Index* (GAI) computations.

This module provides examples of gamut area index calculations for various
light sources.
"""

from pprint import pprint

import colour
from colour.utilities import message_box

message_box("Gamut Area Index Computations")

message_box('Computing the "D65" illuminant "Gamut Area Index (GAI)".')
print(colour.gamut_area_index(colour.SDS_ILLUMINANTS["D65"]))

print("\n")

message_box(
    'Computing the "D65" illuminant "Gamut Area Index" (GAI) with detailed output data.'
)
pprint(colour.gamut_area_index(colour.SDS_ILLUMINANTS["D65"], additional_data=True))

print("\n")

message_box('Computing the "F2" illuminant "Gamut Area Index (GAI)".')
print(colour.gamut_area_index(colour.SDS_ILLUMINANTS["FL2"]))

print("\n")

message_box('Computing the "CIE Standard Illuminant A" "Gamut Area Index (GAI)".')
print(colour.gamut_area_index(colour.SDS_ILLUMINANTS["A"]))
