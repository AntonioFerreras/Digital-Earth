import time
import os
from datetime import datetime
import numpy as np
import taichi as ti
from renderer import Renderer
from lib.volume_rendering_models import planet_r
from lib.math_utils import np_normalize, np_rotate_matrix
import __main__


SCREEN_RES = (1920, 1080)
TARGET_FPS = 30
UP_DIR = (0, 1, 0)
HELP_MSG = '''
====================================================
Camera:
* Drag with your left mouse button to rotate
* Press W/A/S/D/Q/E to move
====================================================
'''

class Camera:
    def __init__(self, window, up):
        self._window = window
        self._lookat_pos = np.array((0.0, 0.0, 0.0))
        self._camera_pos = np.array((-15000000., 0.0, 15000000.))
        self._up = np_normalize(np.array(up))
        self._last_mouse_pos = None

    @property
    def mouse_exclusive_owner(self):
        return True

    def set_up(self, new_up):
        self._up = new_up

    def update_camera(self, elapsed_time):
        res = self._update_by_wasd(elapsed_time)
        res = self._update_by_mouse(elapsed_time) or res
        return res

    def _update_by_mouse(self, elapsed_time):
        win = self._window
        if not self.mouse_exclusive_owner or not win.is_pressed(ti.ui.RMB):
            self._last_mouse_pos = None
            return False
        mouse_pos = np.array(win.get_cursor_pos())
        if self._last_mouse_pos is None:
            self._last_mouse_pos = mouse_pos
            return False
        # Makes camera rotation feels right
        dx, dy = self._last_mouse_pos - mouse_pos
        self._last_mouse_pos = mouse_pos

        out_dir = self._lookat_pos - self._camera_pos
        leftdir = self._compute_left_dir(np_normalize(out_dir))

        scale = 3 # 40*elapsed_time
        rotx = np_rotate_matrix(self._up, dx * scale)
        roty = np_rotate_matrix(leftdir, dy * scale)

        out_dir_homo = np.array(list(out_dir) + [0.0])
        new_out_dir = np.matmul(np.matmul(roty, rotx), out_dir_homo)[:3]
        self._lookat_pos = self._camera_pos + new_out_dir

        return True

    def _compute_cam_r(self):
        return np.sqrt(self._camera_pos[0]*self._camera_pos[0] + 
                                   self._camera_pos[1]*self._camera_pos[1] + 
                                   self._camera_pos[2]*self._camera_pos[2])
    def _update_by_wasd(self, elapsed_time):
        win = self._window
        tgtdir = self.target_dir
        leftdir = self._compute_left_dir(tgtdir)
        lut = [
            ('w', tgtdir),
            ('a', leftdir),
            ('s', -tgtdir),
            ('d', -leftdir),
            (ti.ui.CTRL, -self._up),
            (ti.ui.SPACE, self._up),
        ]
        dir = np.array([0.0, 0.0, 0.0])
        pressed = False
        for key, d in lut:
            if win.is_pressed(key):
                pressed = True
                dir += np.array(d)
        if win.is_pressed('q'):
            pressed = True
            new_up = np_normalize(self._camera_pos)
            self.set_up(new_up)
        if win.is_pressed('e'):
            pressed = True
            new_up = np.array((0.0, 1.0, 0.0))
            self.set_up(new_up)

        if win.is_pressed('i'):
            f = open("config.txt", "w")
            f.write(str(self._camera_pos[0]) + " " + str(self._camera_pos[1]) + " " + str(self._camera_pos[2]) + "\n")
            f.write(str(self._lookat_pos[0]) + " " + str(self._lookat_pos[1]) + " " + str(self._lookat_pos[2]) + "\n")
            f.write(str(self._up[0]) + " " + str(self._up[1]) + " " + str(self._up[2]) + "\n")
            f.close()

        if win.is_pressed('o'):
            f = open("config.txt")
            cam_pos_string = f.readline().split()
            cam_lookat_string = f.readline().split()
            cam_up_string = f.readline().split()
            f.close()

            self._camera_pos[0] = float(cam_pos_string[0])
            self._camera_pos[1] = float(cam_pos_string[1])
            self._camera_pos[2] = float(cam_pos_string[2])

            self._lookat_pos[0] = float(cam_lookat_string[0])
            self._lookat_pos[1] = float(cam_lookat_string[1])
            self._lookat_pos[2] = float(cam_lookat_string[2])

            self._up[0] = float(cam_up_string[0])
            self._up[1] = float(cam_up_string[1])
            self._up[2] = float(cam_up_string[2])

            pressed = True


        if not pressed:
            return False
        dir *= 0.05

        speed = 30.0 * max(min(self._compute_cam_r() - planet_r, planet_r*0.5), 0.0)
        if win.is_pressed(ti.ui.SHIFT):
            speed *= 3.0
        cam_step = dir*speed*elapsed_time
        self._lookat_pos += cam_step
        self._camera_pos += cam_step
        if self._compute_cam_r() < planet_r*1.000:
            self._lookat_pos -= cam_step*2
            self._camera_pos -= cam_step*2 
        
        
        
        return True

    @property
    def position(self):
        return self._camera_pos

    @property
    def look_at(self):
        return self._lookat_pos

    @property
    def target_dir(self):
        return np_normalize(self.look_at - self.position)

    def _compute_left_dir(self, tgtdir):
        cos = np.dot(self._up, tgtdir)
        if abs(cos) > 0.999:
            return np.array([-1.0, 0.0, 0.0])
        return np.cross(self._up, tgtdir)


class EarthViewer:
    def __init__(self):
        ti.init(arch=ti.vulkan, offline_cache=True)
        print(HELP_MSG)
        self.window = ti.ui.Window("Earth Viewer",
                                   SCREEN_RES,
                                   vsync=False)
        self.camera = Camera(self.window, up=UP_DIR)
        self.renderer = Renderer(image_res=SCREEN_RES,
                                 up=UP_DIR)
        self.renderer.set_camera_pos(*self.camera.position)
        if not os.path.exists('screenshot'):
            os.makedirs('screenshot')
        self.renderer.copy_textures()

        print (str(len(self.renderer.crf_names)) + " Camera responses loaded.")

    def start(self):
        canvas = self.window.get_canvas()
        gui = self.window.get_gui()
        spp = 1
        elapsed_time = 1.0

        # GUI
        enable_gui = True
        current_fov = self.renderer.fov[None]
        current_aspect_scale = self.renderer.aspect_scale[None]
        current_exposure = self.renderer.exposure[None]

        selected_crf = 0
        current_gamma = self.renderer.gamma[None]

        current_sun_angle = self.renderer.sun_angle[None]
        current_sun_path_rot = self.renderer.sun_path_rot[None]
        ##########

        while self.window.running:
            should_reset_framebuffer = False

            if self.camera.update_camera(elapsed_time):
                self.renderer.set_camera_pos(*self.camera.position)
                look_at = self.camera.look_at
                self.renderer.set_look_at(*look_at)
                up = self.camera._up
                self.renderer.set_up(*up)
                should_reset_framebuffer = True

            if self.window.is_pressed('i'):
                f = open("config.txt", "a")
                f.write(str(current_fov) + "\n")
                f.write(str(current_aspect_scale) + "\n")
                f.write(str(current_exposure) + "\n")
                f.write(str(selected_crf) + "\n")
                f.write(str(current_gamma) + "\n")
                f.write(str(current_sun_angle) + "\n")
                f.write(str(current_sun_path_rot))
                f.close()

            if self.window.is_pressed('o'):
                f = open("config.txt")
                f.readline()
                f.readline()
                f.readline()
                current_fov = float(f.readline())
                current_aspect_scale = float(f.readline())
                current_exposure = float(f.readline())
                selected_crf = int(f.readline())
                current_gamma = float(f.readline())
                current_sun_angle = float(f.readline())
                current_sun_path_rot = float(f.readline())
                f.close()

            

            t = time.time()
            for _ in range(spp):
                self.renderer.accumulate()
            img = self.renderer.fetch_image()
            if self.window.is_pressed('p'):
                timestamp = datetime.today().strftime('%Y-%m-%d-%H%M%S')
                dirpath = os.getcwd()
                main_filename = os.path.split(__main__.__file__)[1]
                fname = os.path.join(dirpath, 'screenshot', f"{main_filename}-{timestamp}.jpg")
                ti.tools.image.imwrite(img, fname)
                print(f"Screenshot has been saved to {fname}")
            canvas.set_image(img)
            elapsed_time = time.time() - t
            # if elapsed_time * TARGET_FPS > 1:
            #     spp = int(spp / (elapsed_time * TARGET_FPS) - 1)
            #     spp = max(spp, 1)
            # else:
            #     spp += 1

            
            if self.window.is_pressed("g"):
                enable_gui = not enable_gui

            if enable_gui:
                with gui.sub_window("Settings", x=0.025, y=0.025, width=0.25, height=0.3) as g:
                    g.text("Press G to show/hide GUI")

                    g.text("\nWorld")
                    new_sun_angle = np.deg2rad(g.slider_float("Sun angle", np.rad2deg(current_sun_angle), 0.0, 360.0))
                    if new_sun_angle != current_sun_angle:
                        should_reset_framebuffer = True
                        current_sun_angle = new_sun_angle
                    new_sun_path_rot = np.deg2rad(g.slider_float("Sun path rotation", np.rad2deg(current_sun_path_rot), -105.0, 105.0))
                    if new_sun_path_rot != current_sun_path_rot:
                        should_reset_framebuffer = True
                        current_sun_path_rot = new_sun_path_rot
                        self.renderer.sun_path_rot[None] = new_sun_path_rot

                    g.text("\nCamera")
                    new_fov = np.deg2rad(g.slider_float("Verticle FOV", np.rad2deg(current_fov), 1.0, 90.0))
                    if new_fov != current_fov:
                        should_reset_framebuffer = True
                        current_fov = new_fov
                        self.renderer.fov[None] = current_fov

                    new_aspect_scale = g.slider_float("Aspect scale", current_aspect_scale, 0.9, 1.1)
                    if new_aspect_scale != current_aspect_scale:
                        should_reset_framebuffer = True
                        current_aspect_scale = new_aspect_scale
                        self.renderer.aspect_scale[None] = current_aspect_scale
                    
                    new_exposure = g.slider_float("Exposure", current_exposure, -1.0, 8.0)
                    if new_exposure != current_exposure:
                        current_exposure = new_exposure
                        self.renderer.exposure[None] = current_exposure
                    
                    new_crf = g.slider_int("Camera response", selected_crf, 0, len(self.renderer.crf_names)-1)
                    if new_crf != selected_crf:
                        selected_crf = new_crf
                        self.renderer.selected_crf[None] = selected_crf
                    g.text(f"    {self.renderer.crf_names[selected_crf]}")

                    new_gamma = g.slider_float("Gamma", current_gamma, 0.45, 2.2)
                    if new_gamma != current_gamma:
                        current_gamma = new_gamma
                        self.renderer.gamma[None] = current_gamma

            
            self.renderer.sun_angle[None] = current_sun_angle
            self.renderer.sun_path_rot[None] = new_sun_path_rot
            self.renderer.fov[None] = current_fov
            self.renderer.aspect_scale[None] = current_aspect_scale
            self.renderer.exposure[None] = current_exposure
            self.renderer.gamma[None] = current_gamma
            self.renderer.selected_crf[None] = selected_crf

            if should_reset_framebuffer:
                self.renderer.reset_framebuffer()
                
            self.window.show()