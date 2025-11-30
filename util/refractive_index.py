import numpy as np

def refractive_index_noa61(wavelength, a=1.5375, b=0.00829045, c=-0.000211046):
    """ NOA61 dispersion formula
    https://refractiveindex.info/?shelf=other&book=Optical_adhesives&page=Norland_NOA61
    """
    return a + b / (wavelength * 1e6) ** 2 + c / (wavelength * 1e6) ** 4


def refractive_index_glass_bk7(wavelength):
    # dispersion formula
    wavelength_square = (wavelength * 1e6) ** 2
    # B1*λ^2 / (λ^2-C1)
    first = (1.03961212 * wavelength_square) / (wavelength_square - 0.00600069867)
    # B2*λ^2 / (λ^2-C2)
    second = (0.231792344 * wavelength_square) / (wavelength_square - 0.0200179144)
    # B3*λ^2 / (λ^2-C3)
    third = (1.01046945 * wavelength_square) / (wavelength_square - 103.560653)
    return ((first + second + third) + 1) ** 0.5


def refractive_index_glass_ohara_sk1300(wavelength):
    wavelength_square = (wavelength * 1e6) ** 2
    # B1*λ^2 / (λ^2-C1)
    first = (7.44386780E-01 * wavelength_square) / (wavelength_square - 4.95119834E-03)
    # B2*λ^2 / (λ^2-C2)
    second = (3.60198890E-01 * wavelength_square) / (wavelength_square - 1.41130274E-02)
    # B3*λ^2 / (λ^2-C3)
    third = (9.18623119E-01 * wavelength_square) / (wavelength_square - 1.00428510E+02)
    return ((first + second + third) + 1) ** 0.5


MATERIAL_REFRACTIVE_INDEX_FUNCS = {
    "NOA61": refractive_index_noa61,
    "BK7": refractive_index_glass_bk7,
    "SK1300": refractive_index_glass_ohara_sk1300
}


def heightmap_to_phase(height_map, wave_lengths, refractive_index_func):
    """
    Calculates the phase shifts created by a height map with certain refractive index for light with specific wave length.

    Args:
        input_field: Input field.
        height_map: DOE height map.
        wave_lengths: Wavelength list.
        wavelength_to_refractive_index_func:  Refractive index function of the DOE material.

    Returns: Modulated wave field.
    """

    delta_n = refractive_index_func(wave_lengths) - 1
    wave_numbers = 2. * np.pi / wave_lengths
    # wave_numbers = wave_numbers.reshape([1, 1, 1, -1])
    phi = wave_numbers * delta_n * height_map

    return phi

def _phase_to_height_with_material_refractive_idx_func(_phase, _wavelength, _refractive_index_function):
    return _phase / (2 * math.pi / _wavelength) / (_refractive_index_function(_wavelength) - 1)


wavelength = 532e-9
print(refractive_index_glass_bk7(wavelength))
print(heightmap_to_phase(20e-9,532e-9,MATERIAL_REFRACTIVE_INDEX_FUNCS["BK7"]))