# <a name="title">Earth Viewer</a>
Demo shots             |  1080p
:-------------------------:|:-------------------------:
![Earth render 1](screenshot/main.py-2023-12-04-010432.jpg) | ![Earth render 2](screenshot/main.py-2023-12-04-002739.jpg)
![Earth render 1](screenshot/main.py-2023-12-05-002427.jpg) | ![Earth render 2](screenshot/main.py-2023-12-05-003448.jpg)

A renderer that generates highly realistic images of the Earth from outer space. 

Rendering is done with Path Tracing in participating media. 
+ Uses measured data for density and light response (extinction coeficients and scattering functions) of atmospheric gases
  + Rayleigh scatterers: Nitrogen (N2), Oxygen (O2),  Carbon Dioxide (CO2)
  + Mie scatterers: Water vapour
  + Other absorbers: Ozone (O3)
+ Uses spectral rendering to get an accurate representation of colours. 
+ Uses importance sampling in the form of direct light (sun) sampling and phase function pdf.

**Taichi Lang documentation:** https://docs.taichi-lang.org/

## Installation

Make sure your `pip` is up-to-date:

```bash
Python:
>>> pip3 install pip --upgrade
```

Assume you have a Python 3 environment, simply run:

```bash
>>> pip3 install -r requirements.txt
```

to install the dependencies of the renderer.

Then in the Earth Viewer folder, run
```bash
>>> python3 main.py
```


## Controls

+ Drag with your right mouse button to rotate the camera.
+ Press `W/A/S/D` to move the camera.
  + Press 'Q' to rotate the camera to Earth's surface
  + Press 'E' to reset camera rotation
  + Press SPACE to go up
  + Press CTRL to go down
+ Press `P` to save a screenshot.

