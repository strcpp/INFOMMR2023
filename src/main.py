import moderngl_window as glw
import moderngl as gl
from render.shaders import Shaders
from render.mesh import Mesh
from scenes.basic_scene import BasicScene
import pathlib
import numpy as np
import imgui
from moderngl_window.integrations.imgui import ModernglWindowRenderer
from typing import Any, Tuple
from moderngl_window.text.bitmapped import TextWriter2D


class App(glw.WindowConfig):
    """
    Main glw App.
    """
    title = "Multimedia retrieval"
    gl_version = (3, 3)
    window_size = (1600, 800)
    aspect_ratio = None
    resource_dir = (pathlib.Path(__file__).parent.parent / "resources").resolve()
    samples = 16

    def __init__(self, *args: Tuple[Any], **kwargs: Any) -> None:
        """
        Constructor.
        :param args: Positional arguments.
        :param kwargs: Keyword arguments.
        """
        super().__init__(*args, **kwargs)  # type: ignore
        self.camera = glw.scene.camera.OrbitCamera(aspect_ratio=self.wnd.aspect_ratio)
        self.camera.zoom_sensitivity = 0.05
        # self.camera.mouse_sensitivity = 0.75
        # self.camera.zoom_state(5)
        # self.camera.set_position(0.04484549, 0.79429578, 1.83496133)

        self.mouse_pressed = False
        self.mouse_button = 0
        self.mpos = (0, 0)
        self.mdelta = (0, 0)

        # initialize all assets
        Shaders.instance(self)
        Mesh.instance(self)

        imgui.create_context()

        self.imgui = ModernglWindowRenderer(self.wnd)
        self.writer = TextWriter2D()

        self.fps_dims = (10, self.window_size[1] - 10)

        self.scene = BasicScene(self)
        self.scene.load()

    def render(self, time: float, frame_time: float) -> None:
        """
        Main glw render function.
        :param time: Elapsed time.
        :param frame_time: Time passed after the previous frame.
        """

        self.ctx.enable(int(str(gl.DEPTH_TEST)))

        self.ctx.clear(color=(0.09, 0.12, 0.23, 0))
        self.scene.update(frame_time)

        self.scene.render()

        if frame_time != 0:
            self.writer.text = "{:.2f}".format(1.0 / frame_time)
        else:
            self.writer.text = "0"

        self.writer.draw(self.fps_dims, size=20)
        # print(self.camera.position)
        # print(self.camera.pitch, " ", self.camera.yaw)


    def key_event(self, key: int, action: str, modifiers: glw.context.base.keys.KeyModifiers) -> None:
        """
        Key even method.
        :param key: Key code or identifier associated with the key event.
        :param action: Action performed on the key (e.g., "press", "release").
        :param modifiers: Key modifiers (e.g., Shift, Control, Alt).
        """
        keys = self.wnd.keys
        if action == keys.ACTION_PRESS and key == keys.ESCAPE:
            self.wnd.close()
        self.imgui.key_event(key, action, modifiers)
        self.scene.key_event(key, action)

    def mouse_drag_event(self, x: int, y: int, dx: int, dy: int) -> None:
        """
        Mouse drag event method.
        :param x: Mouse coordinates on the x-axis.
        :param y: Mouse coordinates on the y-axis.
        :param dx: Coordinate change on the x-axis since the last mouse drag event.
        :param dy: Coordinate change on the y-axis since the last mouse drag event.
        """
        if not imgui.get_io().want_capture_mouse:
            # pan camera, orbit camera class does not offer this for some reason...
            if self.mouse_button == 3:
                view_matrix = self.camera.matrix
                right = np.array(view_matrix.c1)
                up = np.array(view_matrix.c2)

                translation = dx * 0.01 * right + dy * 0.01 * up

                self.camera.target = [
                    self.camera.target[0] + translation[0],
                    self.camera.target[1] + translation[1],
                    self.camera.target[2] + translation[2]
                ]
            else:
                self.camera.rot_state(dx, dy)

        self.imgui.mouse_drag_event(x, y, dx, dy)

    def mouse_scroll_event(self, x_offset: float, y_offset: float) -> None:
        """
        Mouse scroll event method.
        :param x_offset: Horizontal scroll offset.
        :param y_offset: Vertical scroll offset.
        """
        self.camera.zoom_state(y_offset)
        self.imgui.mouse_scroll_event(x_offset, y_offset)

    def resize(self, width: int, height: int) -> None:
        """
        Window resize method.
        :param width: Resized window width.
        :param height: Resized window height.
        """
        self.camera.projection.update(aspect_ratio=self.wnd.aspect_ratio)
        self.imgui.resize(width, height)
        self.fps_dims = (10, height - 10)

    def mouse_press_event(self, x: int, y: int, button: int) -> None:
        """
        Mouse press event method.
        :param x: Mouse coordinates on the x-axis.
        :param y: Mouse coordinates on the y-axis.
        :param button: Pressed button code.
        """
        self.mouse_pressed = True
        self.mouse_button = button
        self.imgui.mouse_press_event(x, y, button)

    def mouse_release_event(self, x: int, y: int, button: int) -> None:
        """
        Mouse release event method.
        :param x: Mouse coordinates on the x-axis.
        :param y: Mouse coordinates on the y-axis.
        :param button: Released button code.
        """
        self.mouse_pressed = False
        self.mouse_button = None
        self.imgui.mouse_release_event(x, y, button)

    def mouse_position_event(self, x: int, y: int, dx: int, dy: int) -> None:
        """
        Mouse position event method.
        :param x: Mouse coordinates on the x-axis.
        :param y: Mouse coordinates on the y-axis.
        :param dx: Coordinate change on the x-axis since the last mouse drag event.
        :param dy: Coordinate change on the y-axis since the last mouse drag event.
        """
        self.mpos = (x, y)
        self.mdelta = (dx, dy)
        self.imgui.mouse_position_event(x, y, dx, dy)

    def unicode_char_entered(self, char: str) -> None:
        """
        Unicode character entered event method.
        :param char: Entered Unicode character.
        """
        self.imgui.unicode_char_entered(char)


if __name__ == '__main__':
    App.run()
