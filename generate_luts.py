import taichi as ti
import numpy as np
from lib.math_utils import *
from lib.sampling import *
from lib.colour import *
from lib.textures import *
import lib.volume_rendering_models as volume
from lib.parameters import PathParameters, SceneParameters
from lib.geometry import *
from lib.ray_functions import *
from PIL import Image
from lib.lut_sampling import *

ti.init(arch=ti.gpu)

transmittance_lut = ti.field(dtype=ti.f32, shape=TRANS_LUT_RES)

O3_crossec_LUT_buff = ti.field(dtype=ti.f32, shape=(O3_CROSSEC_LUT_RES))
with open(O3_CROSSEC_LUT_FILE, 'rb') as file:
    load_data = np.fromfile(file, dtype=np.float32, count=O3_CROSSEC_LUT_RES)
data_array = np.zeros(shape=(O3_CROSSEC_LUT_RES), dtype=np.float32)
for x in range (0, O3_CROSSEC_LUT_RES):
    data_array[x] = load_data[x]
O3_crossec_LUT_buff.from_numpy(data_array)



@ti.kernel
def compute_transmittance_lut(o3_crossec_buff: ti.template()):
    for x, y, z in transmittance_lut:
        xyz = ti.Vector([ti.cast(x, ti.f32)/TRANS_LUT_RES[0], 
                         ti.cast(y, ti.f32)/TRANS_LUT_RES[1], 
                         ti.cast(z, ti.f32)/TRANS_LUT_RES[2]])
        # cos_theta = pow(xyz.x, 2.0)*1.19 - 0.19
        # h = volume.atmos_height * (xyz.y)
        cos_theta, h = uv_to_mu_r(xyz.xy)
        wavelength = 390.0 + xyz.z*441.0

        theta = acos(cos_theta)
        sin_theta = sin(theta)

        ray_pos = vec3(0.0, volume.planet_r + h, 0.0)
        ray_dir = vec3(sin_theta, cos_theta, 0.0)

        extinctions = vec3(0.0, 0.0, 0.0)
        extinctions.x = volume.spectra_extinction_rayleigh(wavelength)
        extinctions.y = volume.spectra_extinction_mie(wavelength)
        extinctions.z = volume.spectra_extinction_ozone(wavelength, o3_crossec_buff)
        
        transmittance = ray_march_transmittance(ray_pos, ray_dir, extinctions, 256)

        transmittance_lut[x, y, z] = transmittance

compute_transmittance_lut(O3_crossec_LUT_buff)

array = transmittance_lut.to_numpy()
array.tofile('LUT/transmittance_lut.dat')

# Read from the newly created file into a new array
# read_array = np.fromfile('LUT/transmittance_lut.dat', dtype=np.float32)
# read_array = read_array.reshape(transmittance_lut.shape)

# Verify that the contents after reading match
# if read_array.shape == transmittance_lut.shape:
#     difference = np.abs(read_array - transmittance_lut.to_numpy())
#     if np.all(difference < 0.0001):
#         print("The contents match.")
#     else:
#         print("The contents do not match.")
# else:
#     print("The shapes do not match.")


# Save the slice as a PNG image
# Take the x,y slice at z = 200

slice_000 = array[:, :, 000]
image = Image.fromarray((slice_000 * 255).astype(np.uint8))
image.save('slice_000.png')

slice_200 = array[:, :, 200]
image = Image.fromarray((slice_200 * 255).astype(np.uint8))
image.save('slice_200.png')

slice_400 = array[:, :, 400]
image = Image.fromarray((slice_400 * 255).astype(np.uint8))
image.save('slice_400.png')