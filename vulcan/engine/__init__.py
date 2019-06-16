import sys
from ctypes import c_float, c_int

import glfw
import numpy as np

from OpenGL.GL import *
from PIL import Image
from abc import ABCMeta, abstractmethod
from vulcan.engine import utils
from vulcan.engine.utils import Matrix4f, Material


class Display:
    width = None
    height = None
    window = None

    def __init__(self):
        pass

    @staticmethod
    def init(width=1200, height=720, title='Python-opengl'):
        # type: (int, int, str) -> None
        """
        Initializes a window.
        Args:
            width: Width of the window
            height: Height of the window
            title: Title of the window

        Returns:
            object: The window
        """
        Display.width = width
        Display.height = height

        if not glfw.init():
            return

        Display.add_window_hints()
        Display.window = glfw.create_window(width, height, title, None, None)
        if not Display.window:
            glfw.terminate()
            return

        glfw.make_context_current(Display.window)

        return Display.window

    @staticmethod
    def add_window_hints():
        # type: () -> None
        """
        Adds window hints.
        Returns:
            None
        """
        glfw.window_hint(glfw.SAMPLES, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    @staticmethod
    def window_closed():
        # type: () -> bool
        """
        Checks if window is closed.
        Returns:
            bool: True if window is closed, False otherwise.
        """
        return glfw.window_should_close(Display.window)

    @staticmethod
    def swap_buffers():
        # type: () -> None
        """
        Swaps the front and back buffers
        Returns:
            None
        """
        glfw.swap_buffers(Display.window)

    @staticmethod
    def poll_events():
        # type: () -> None
        """
        Polls for and process events.
        Returns:
            None
        """
        glfw.poll_events()

    @staticmethod
    def terminate():
        # type: () -> None
        """
        Terminates the window
        Returns:
            None
        """
        glfw.terminate()


class Renderer:
    def __init__(self, shader=None):
        self.shader = shader
        # shader.start()
        # shader.load_projection_matrix(utils.create_projection_matrix(Display.width, Display.height))
        # shader.stop()
        # self.shader = shader

    # noinspection PyMethodMayBeStatic
    def prepare(self):
        # type: () -> None
        """
        Prepares the window before rendering.
        Returns:
            None
        """
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0, 0, 0, 1)

    def render(self, material, transformation_matrix=None):
        # type: (utils.Material, Matrix4f) -> None
        """
        Renders the material on the screen.
        Args:
            material: The material of the entity to be rendered.
            transformation_matrix: The transformation matrix of the entity being rendered.

        Returns:
            None
        """

        self.shader.load_transformation_matrix(transformation_matrix)

        glBindVertexArray(material.vao_id)

        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)

        glActiveTexture(GL_TEXTURE0)  # Activating the texture bank 0. sampler2D accesses this texture bank by default.
        glBindTexture(GL_TEXTURE_2D, material.tex_id)  # binding the texture to be rendered to the texture bank 0

        glDrawElements(GL_TRIANGLES, material.vertex_count, GL_UNSIGNED_INT, None)

        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)

        glBindVertexArray(0)


class Loader:

    __vaos = []
    __vbos = []
    __textures = []

    __indices = np.array([
        0, 1, 3,
        3, 1, 2
    ], dtype=np.uint32)

    __texCoords = np.array([
        0, 0,
        0, 1,
        1, 1,
        1, 0
    ], dtype=np.float32)

    def __init__(self):
        pass

    @staticmethod
    def load_material(sx, sy, texture_path=None):
        # type: (int, int, str) -> Material
        """
        Loads the vertex, indices and texture data to a VAO, and returns a :class:`Material`
        object.
        Args:
            sx: Width of the entity on which the texture will be applied
            sy: Height of the entity on which the texture will be applied
            texture_path: The path to the texture

        Returns:
            :class:`Material` object.
        """
        vertices = np.array([
            -sx/2, sy/2,
            -sx/2, -sy/2,
            sx/2, -sy/2,
            sx/2, sy/2
        ], dtype=np.float32)

        vao = Loader.__create_vao()  # type: int
        tex = 0
        Loader.__bind_vao(vao)
        Loader.__bind_indices_buffer(Loader.__indices)

        if texture_path is not None:
            Loader.__store_data_in_attribute_list(attribute_number=1, coord_dim=2, data=Loader.__texCoords)
            tex = Loader.__load_texture(texture_path)  # type: int

        Loader.__store_data_in_attribute_list(attribute_number=0, coord_dim=2, data=vertices)
        Loader.__unbind_vao()

        return utils.Material(vao, len(Loader.__indices), tex)

    @staticmethod
    def clean_up():
        """
        Deletes the created VAOs, VBOs and textures.
        Returns:
            None
        """
        for vao in Loader.__vaos:
            glDeleteVertexArrays(1, vao)

        for vbo in Loader.__vbos:
            glDeleteBuffers(1, vbo)

        for tex in Loader.__textures:
            glDeleteTextures(1, tex)

    @staticmethod
    def __create_vao():
        """
        creates a VAO, tracks it and returns it.
        Returns:
            id of the created VAO
        """
        vao = glGenVertexArrays(1)  # type: int
        Loader.__vaos.append(vao)
        return vao

    @staticmethod
    def __bind_vao(vao):
        """
        Binds the given VAO
        Args:
            vao: id of the VAO to be bound

        Returns:
            None

        """
        glBindVertexArray(vao)

    @staticmethod
    def __unbind_vao():
        """
        unbinds the currently bound VAO
        Returns:
            None
        """
        glBindVertexArray(0)

    @staticmethod
    def __bind_indices_buffer(indices):
        """
        creates a VBO, binds it, and stores the indices data in it. Used for storing
        indices data only.
        Args:
            indices: The indices data

        Returns:
            None
        """
        vbo = glGenBuffers(1)
        Loader.__vbos.append(vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices, GL_STATIC_DRAW)

    @staticmethod
    def __store_data_in_attribute_list(attribute_number, coord_dim, data):
        """
        creates a VBO, binds it, stores data in it and unbinds it.
        Args:
            attribute_number: The attribute number of the data being stored
            coord_dim: Number of dimensions of the coordinates
            data: The data to be stored

        Returns:
            None
        """

        vbo = glGenBuffers(1)

        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, data, GL_STATIC_DRAW)
        glVertexAttribPointer(attribute_number, coord_dim, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        Loader.__vbos.append(vbo)

    @staticmethod
    def __load_texture(path):
        """
        Loads the texture from file
        Args:
            path: Path of the image file

        Returns:
            texture id
        """

        img = Image.open(path)
        img_data = np.fromstring(img.tobytes(), np.uint8)

        tex = glGenTextures(1)  # type: int
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glBindTexture(GL_TEXTURE_2D, tex)

        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, -1)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.size[0], img.size[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        glBindTexture(GL_TEXTURE_2D, 0)

        Loader.__textures.append(tex)
        return tex


class ShaderProgram(object):
    __metaclass__ = ABCMeta

    def __init__(self, vertex_shader, fragment_shader):
        # type: (str, str) -> None

        self.program = glCreateProgram()  # type: int
        self.vs = ShaderProgram.__load(vertex_shader, GL_VERTEX_SHADER)
        self.fs = ShaderProgram.__load(fragment_shader, GL_FRAGMENT_SHADER)

        glAttachShader(self.program, self.vs)
        glAttachShader(self.program, self.fs)

        self.bind_attributes()
        glLinkProgram(self.program)
        glValidateProgram(self.program)
        self.get_all_uniform_locations()
        self.connect_texture_units()

    def get_uniform_location(self, var_name):
        # type: (str) -> int
        """
        Returns the location of the uniform variable in the shader program.
        Args:
            var_name: The name of the uniform variable in the shader program.

        Returns:
            int: The location of the uniform variable.
        """
        return glGetUniformLocation(self.program, var_name)

    def bind_attribute(self, attribute_number, variable):
        # type: (int, str) -> None
        """
        Binds the attribute (variable) used in the shader program to the attribute
        location in the currently bound VAO.
        Args:
            attribute_number: The attribute location of the VAO to be bound.
            variable: the name of the variable to be bound.

        Returns:
            None
        """
        glBindAttribLocation(self.program, attribute_number, variable)

    @staticmethod
    def load_float(location, value):
        # type: (int, float) -> None
        """
        Loads a float value to the uniform variable.
        Args:
            location: The uniform variable to be loaded.
            value: The value to load.

        Returns:
            None
        """
        glUniform1f(location, value)

    @staticmethod
    def load_vector2f(location, vector):
        # type: (int, utils.Vector2f) -> None
        """
        Loads a 2D Vector value to the uniform variable.
        Args:
            location: The location of the uniform variable to be loaded.
            vector: The vector to be loaded.

        Returns:
            None
        """
        glUniform2f(location, vector.x, vector.y)

    @staticmethod
    def load_vector3f(location, vector):
        # type: (int, utils.Vector3f) -> None
        """
        Loads a 3D Vector value to the uniform variable.
        Args:
            location: The location of the uniform variable to be loaded.
            vector: The vector to be loaded.

        Returns:
            None
        """
        glUniform3f(location, vector.x, vector.y, vector.z)

    @staticmethod
    def load_vector4f(location, vector):
        # type: (int, utils.Vector4f) -> None
        """
        Loads a 4D Vector value to the uniform variable.
        Args:
            location: The location of the uniform variable to be loaded.
            vector: The vector to be loaded.

        Returns:
            None
        """
        glUniform4f(location, vector.x, vector.y, vector.z, vector.w)

    @staticmethod
    def load_boolean(location, value):
        # type: (int, bool) -> None
        """
        Loads a boolean value to the uniform variable.
        Args:
            location: The location of the uniform variable to be loaded.
            value: The value to be loaded.

        Returns:
            None
        """
        glUniform1f(location, 1 if value else 0)

    @staticmethod
    def load_int(location, value):
        # type: (int, int) -> None
        """
        Loads an int value to the uniform variable.
        Args:
            location: The location of the uniform variable to be loaded.
            value: The value to be loaded.

        Returns:
            None
        """
        glUniform1i(location, value)

    @staticmethod
    def load_matrix(location, matrix):
        # type: (int, utils.Matrix4f) -> None
        """
        Loads a matrix value to the uniform variable.
        Args:
            location: The location of the uniform variable to be loaded.
            matrix: The matrix to be loaded.

        Returns:
            None
        """
        glUniformMatrix4fv(location, 1, GL_FALSE, matrix.data)

    @staticmethod
    def __load(path, tp):
        # type: (str, int) -> int
        """
        Loads the shader source code, creates a shader and attaches the shader source to it.
        Compiles the shader source code, exits with error if compilation fails.

        Args:
            path: Path to the shader source.
            tp: type of shader to compile.

        Returns:
            the shader id

        """
        source = "".join(open(path).readlines())

        shader = glCreateShader(tp)  # type: int
        glShaderSource(shader, source)
        glCompileShader(shader)

        if glGetShaderiv(shader, GL_COMPILE_STATUS) == GL_FALSE:
            print(glGetShaderInfoLog(shader, 500))
            print("Could not compile shader")
            sys.exit(-1)

        return shader

    def start(self):
        # type: () -> None
        """
        Start using the shader program.
        Returns:
            None
        """
        glUseProgram(self.program)

    # noinspection PyMethodMayBeStatic
    def stop(self):
        # type: () -> None
        """
        Stop using the shader program.
        Returns:
            None
        """
        glUseProgram(0)

    def clean_up(self):
        # type: () -> None
        """
        Stops the shader program, detaches the shaders and deletes it.
        Returns:
            None
        """
        self.stop()
        glDetachShader(self.program, self.vs)
        glDetachShader(self.program, self.fs)
        glDeleteProgram(self.program)
        glDeleteShader(self.vs)
        glDeleteShader(self.fs)

    @abstractmethod
    def bind_attributes(self):
        # type: () -> None
        pass

    @abstractmethod
    def get_all_uniform_locations(self):
        # type: () -> None
        pass

    @abstractmethod
    def connect_texture_units(self):
        # type: () -> None
        pass
