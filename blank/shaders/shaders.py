from vulcan.engine import ShaderProgram, Matrix4f


class StaticShader(ShaderProgram):
    vertex_file = '/home/ouroboros/Projects/python-opengl/blank/shaders/vertex_shader.vert'
    fragment_file = '/home/ouroboros/Projects/python-opengl/blank/shaders/fragment_shader.frag'

    def __init__(self):
        super(StaticShader, self).__init__(StaticShader.vertex_file, StaticShader.fragment_file)

    def get_all_uniform_locations(self) -> None:
        pass

    def connect_texture_units(self) -> None:
        pass

    def bind_attributes(self) -> None:
        self.bind_attribute(0, 'position')
        self.bind_attribute(1, 'texture_coords')

    def load_transformation_matrix(self, matrix: Matrix4f) -> None:
        """
        Loads the transformation matrix to the shader program.
        Args:
            matrix (Matrix4f): The transformation matrix

        Returns:
            None
        """
        transform_var = self.get_uniform_location('transform')
        self.load_matrix(transform_var, matrix)
