TEXTURE_QUALITY = 2 
TEX_RES_4K = (3840, 1920)
TEX_RES_8K = (8100, 4050)
TEX_RES_10K = (10800, 5400)
TEX_RES_16K = (16200, 8100)
TEX_RES_21K = (21600, 10800)
CIE_LUT_RES = (441, 2)
O3_CROSSEC_LUT_RES = 441

ALBEDO_4K = 'textures/earth_color_4K.png'
ALBEDO_10K = 'textures/earth_color_10K.png'
ALBEDO_21K = 'textures/earth_color_21K.png'
TOPOGRAPHY_4K = 'textures/topography_4K.png'
TOPOGRAPHY_10K = 'textures/topography_10K.png'
TOPOGRAPHY_21K = 'textures/topography_21K.png'
OCEAN_4K = 'textures/earth_landocean_4K.png'
OCEAN_8K = 'textures/earth_landocean_8K.png'
OCEAN_16K = 'textures/earth_landocean_16K.png'
CLOUDS_4K = 'textures/earth_clouds_4K.png'
CLOUDS_8K = 'textures/earth_clouds_8K.png'
CLOUDS_21K = 'textures/earth_clouds_21K.png'
BATHYMETRY_4K = 'textures/earth_bathymetry_4k.png'
BATHYMETRY_10K = 'textures/earth_bathymetry_10k.png'
BATHYMETRY_21K = 'textures/earth_bathymetry_21k.png'
EMISSIVE_4K = 'textures/earth_nightlights_4K.png'
EMISSIVE_10K = 'textures/earth_nightlights_10K.png'
EMISSIVE_21K = 'textures/earth_nightlights_21K.png'

CIE_LUT_FILE = 'LUT/CIE.dat'
SRGB2SPEC_LUT_FILE = 'LUT/srgb2spec.dat'
O3_CROSSEC_LUT_FILE = 'LUT/ozone_cross_section.dat'


ALBEDO_TEX_FILE = ALBEDO_4K
ALBEDO_TEX_RES = TEX_RES_4K
TOPOGRAPHY_TEX_FILE = TOPOGRAPHY_4K
TOPOGRAPHY_TEX_RES = ALBEDO_TEX_RES
OCEAN_TEX_FILE = OCEAN_4K
OCEAN_TEX_RES = TEX_RES_4K
CLOUDS_TEX_FILE = CLOUDS_4K
BATHYMETRY_TEX_FILE = BATHYMETRY_4K
BATHYMETRY_TEX_RES = TEX_RES_4K
EMISSIVE_TEX_FILE = EMISSIVE_4K
EMISSIVE_TEX_RES = TEX_RES_4K
STARS_TEX_FILE = 'textures/stars_8K.jpg'
STARS_TEX_RES = TEX_RES_8K


if TEXTURE_QUALITY == 1:
    ALBEDO_TEX_FILE = ALBEDO_10K
    ALBEDO_TEX_RES = TEX_RES_10K
    TOPOGRAPHY_TEX_FILE = TOPOGRAPHY_10K
    TOPOGRAPHY_TEX_RES = TEX_RES_10K
    OCEAN_TEX_FILE = OCEAN_8K
    OCEAN_TEX_RES = TEX_RES_8K
    CLOUDS_TEX_FILE = CLOUDS_8K
    CLOUDS_TEX_RES = TEX_RES_8K
    BATHYMETRY_TEX_FILE = BATHYMETRY_10K
    BATHYMETRY_TEX_RES = TEX_RES_10K
    EMISSIVE_TEX_FILE = EMISSIVE_10K
    EMISSIVE_TEX_RES = TEX_RES_10K
    STARS_TEX_FILE = 'textures/stars_16K.png'
    STARS_TEX_RES = TEX_RES_16K

elif TEXTURE_QUALITY == 2:
    ALBEDO_TEX_FILE = ALBEDO_21K
    ALBEDO_TEX_RES = TEX_RES_21K
    TOPOGRAPHY_TEX_FILE = TOPOGRAPHY_21K
    TOPOGRAPHY_TEX_RES = TEX_RES_21K
    OCEAN_TEX_FILE = OCEAN_16K
    OCEAN_TEX_RES = TEX_RES_16K
    CLOUDS_TEX_FILE = CLOUDS_21K
    CLOUDS_TEX_RES = TEX_RES_21K
    BATHYMETRY_TEX_FILE = BATHYMETRY_21K
    BATHYMETRY_TEX_RES = TEX_RES_21K
    EMISSIVE_TEX_FILE = EMISSIVE_21K
    EMISSIVE_TEX_RES = TEX_RES_21K
    STARS_TEX_FILE = 'textures/stars_16K.png'
    STARS_TEX_RES = TEX_RES_16K
