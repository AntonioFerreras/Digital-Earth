# <a name="title">Earth Viewer</a>

<p align="center">
<img src="demo.jpg" width="75%"></img>
</p>

A renderer that generates highly realistic images of the Earth from outer space. 

Rendering is done with Path Tracing in participating media. 
+ Uses importance sampling in the form of direct light (sun) sampling and phase function pdf.
+ Uses spectral rendering to get an accurate representation of colours. 

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

to install the dependencies of the voxel renderer.

## Controls

+ Drag with your left mouse button to rotate the camera.
+ Press `W/A/S/D/Q/E` to move the camera.
+ Press `P` to save a screenshot.

