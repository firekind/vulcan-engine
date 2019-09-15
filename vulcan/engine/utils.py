import numpy as np


def create_transformation_matrix(translation: 'Vector3f', rot: 'Vector3f', scale: 'Vector3f') -> 'Matrix4f':
    """
    Creates a transformation matrix.
    Args:
        translation: Translation to be applied.
        rot: Rotation to be applied.
        scale: Scale to be applied.

    Returns:
        Matrix4f: The transformation matrix
    """

    matrix = Matrix4f()

    Matrix4f.apply_translation(vector=translation, src=matrix, dest=matrix)
    Matrix4f.apply_rotation(rot.x, Vector3f(1, 0, 0), matrix, matrix)
    Matrix4f.apply_rotation(rot.y, Vector3f(0, 1, 0), matrix, matrix)
    Matrix4f.apply_rotation(rot.z, Vector3f(0, 0, 1), matrix, matrix)
    Matrix4f.apply_scale(scale, matrix, matrix)

    return matrix


def create_projection_matrix(width: int, height: int) -> 'Matrix4f':
    """
    Creates a projection matrix.
    Returns:
        Matrix4f: Projection Matrix.
    """

    RIGHT_PLANE = float(width)
    LEFT_PLANE = 0.
    NEAR_PLANE = -1.
    FAR_PLANE = 1.
    TOP_PLANE = float(height)
    BOTTOM_PLANE = 0.

    matrix = Matrix4f()

    matrix.m00 = 2. / (RIGHT_PLANE - LEFT_PLANE)
    matrix.m11 = 2. / (TOP_PLANE - BOTTOM_PLANE)
    matrix.m22 = -2. / (FAR_PLANE - NEAR_PLANE)
    matrix.m33 = 1
    matrix.m30 = -((RIGHT_PLANE + LEFT_PLANE) / (RIGHT_PLANE - LEFT_PLANE))
    matrix.m31 = -((TOP_PLANE + BOTTOM_PLANE) / (TOP_PLANE - BOTTOM_PLANE))
    matrix.m32 = -((FAR_PLANE + NEAR_PLANE) / (FAR_PLANE - NEAR_PLANE))

    return matrix


class Material:
    def __init__(self, vao: int, vertex_count: int, tex: int = None) -> None:
        self.vao = vao
        self.vertex_count = vertex_count
        self.tex = tex

    @property
    def vao_id(self) -> int:
        return self.vao

    @property
    def tex_id(self) -> int:
        return self.tex


class Matrix4f:
    def __init__(self) -> None:
        self.data = np.identity(4, dtype=np.float32)

    def set_identity(self) -> None:
        """
        Sets the matrix to the identity matrix.
        Returns:
            None
        """
        self.data = np.identity(4, dtype=np.float32)

    def translate(self, vector: 'Vector3f', dest: 'Matrix4f' = None) -> 'Matrix4f':
        """
        Translates the matrix.
        Args:
            vector: Translation vector (3x1)
            dest: Destination matrix (4x4)

        Returns:
            Matrix4f: Destination matrix (4x4)
        """
        if dest is None:
            return Matrix4f.apply_translation(vector=vector, src=self, dest=self)
        return Matrix4f.apply_translation(vector=vector, src=self, dest=dest)

    @staticmethod
    def apply_translation(vector: 'Vector3f', src: 'Matrix4f', dest: 'Matrix4f' = None) -> 'Matrix4f':
        """
        Translates the vulcan matrix.
        Args:
            vector: Translation vector (3x1)
            src: Source matrix (4x4)
            dest: Destination matrix (4x4)

        Returns:
            Matrix4f: Destination matrix (4x4)
        """
        if dest is None:
            dest = Matrix4f()

        dest.m30 += src.m00 * vector.x + src.m10 * vector.y + src.m20 * vector.z
        dest.m31 += src.m01 * vector.x + src.m11 * vector.y + src.m21 * vector.z
        dest.m32 += src.m02 * vector.x + src.m12 * vector.y + src.m22 * vector.z
        dest.m33 += src.m03 * vector.x + src.m13 * vector.y + src.m23 * vector.z

        return dest

    def rotate(self, angle: float, axis: 'Vector3f', dest: 'Matrix4f' = None) -> 'Matrix4f':
        """
        Applies rotation transformation to the matrix.
        Args:
            angle: Angle to rotate by.
            axis: Vector specifying the axis (3x1)
            dest: Destination Matrix (4x4)

        Returns:
            Matrix4f: Destination Matrix (4x4)
        """
        if dest is None:
            return Matrix4f.apply_rotation(angle=angle, axis=axis, src=self, dest=self)
        return Matrix4f.apply_rotation(angle=angle, axis=axis, src=self, dest=dest)

    @staticmethod
    def apply_rotation(angle: float, axis: 'Vector3f', src: 'Matrix4f', dest: 'Matrix4f' = None):
        """
        Applies rotation transformation to the vulcan matrix.
        Args:
            angle: Angle to rotate by.
            axis: Vector specifying the axis (3x1)
            src: Source Matrix (4x4)
            dest: Destination Matrix (4x4)

        Returns:
            Matrix4f: Destination Matrix (4x4)
        """
        if dest is None:
            dest = Matrix4f()

        c = np.cos(angle)
        s = np.sin(angle)
        oneminusc = 1.0 - c
        xy = axis.x * axis.y
        yz = axis.y * axis.z
        xz = axis.x * axis.z
        xs = axis.x * s
        ys = axis.y * s
        zs = axis.z * s
        f00 = axis.x * axis.x * oneminusc + c
        f01 = xy * oneminusc + zs
        f02 = xz * oneminusc - ys
        f10 = xy * oneminusc - zs
        f11 = axis.y * axis.y * oneminusc + c
        f12 = yz * oneminusc + xs
        f20 = xz * oneminusc + ys
        f21 = yz * oneminusc - xs
        f22 = axis.z * axis.z * oneminusc + c
        t00 = src.m00 * f00 + src.m10 * f01 + src.m20 * f02
        t01 = src.m01 * f00 + src.m11 * f01 + src.m21 * f02
        t02 = src.m02 * f00 + src.m12 * f01 + src.m22 * f02
        t03 = src.m03 * f00 + src.m13 * f01 + src.m23 * f02
        t10 = src.m00 * f10 + src.m10 * f11 + src.m20 * f12
        t11 = src.m01 * f10 + src.m11 * f11 + src.m21 * f12
        t12 = src.m02 * f10 + src.m12 * f11 + src.m22 * f12
        t13 = src.m03 * f10 + src.m13 * f11 + src.m23 * f12

        dest.m20 = src.m00 * f20 + src.m10 * f21 + src.m20 * f22
        dest.m21 = src.m01 * f20 + src.m11 * f21 + src.m21 * f22
        dest.m22 = src.m02 * f20 + src.m12 * f21 + src.m22 * f22
        dest.m23 = src.m03 * f20 + src.m13 * f21 + src.m23 * f22
        dest.m00 = t00
        dest.m01 = t01
        dest.m02 = t02
        dest.m03 = t03
        dest.m10 = t10
        dest.m11 = t11
        dest.m12 = t12
        dest.m13 = t13

        return dest

    def scale(self, vector: 'Vector3f', dest: 'Matrix4f' = None) -> 'Matrix4f':
        """
       Applies scale transformation on the matrix and stores result in dest.
       Args:
           vector: The scale vector
           dest: The destination matrix (4x4)

       Returns:
           Matrix4f: Destination matrix
       """
        if dest is None:
            return Matrix4f.apply_scale(vector=vector, src=self, dest=self)
        return Matrix4f.apply_scale(vector=vector, src=self, dest=dest)

    @staticmethod
    def apply_scale(vector: 'Vector3f', src: 'Matrix4f', dest: 'Matrix4f' = None) -> 'Matrix4f':
        """
        Applies scale transformation on the vulcan matrix and stores result in dest.
        Args:
            vector: The scale vector
            src: The source matrix (4x4)
            dest: The destination matrix (4x4)

        Returns:
            Matrix4f: Destination matrix
        """
        if dest is None:
            dest = Matrix4f()

        dest.m00 = src.m00 * vector.x
        dest.m01 = src.m01 * vector.x
        dest.m02 = src.m02 * vector.x
        dest.m03 = src.m03 * vector.x
        dest.m10 = src.m10 * vector.y
        dest.m11 = src.m11 * vector.y
        dest.m12 = src.m12 * vector.y
        dest.m13 = src.m13 * vector.y
        dest.m20 = src.m20 * vector.z
        dest.m21 = src.m21 * vector.z
        dest.m22 = src.m22 * vector.z
        dest.m23 = src.m23 * vector.z
        return dest

    @property
    def m00(self) -> np.float32:
        return self.data[0][0]

    @m00.setter
    def m00(self, value) -> None:
        self.data[0][0] = value

    @property
    def m01(self) -> np.float32:
        return self.data[0][1]

    @m01.setter
    def m01(self, value) -> None:
        self.data[0][1] = value

    @property
    def m02(self) -> np.float32:
        return self.data[0][2]

    @m02.setter
    def m02(self, value) -> None:
        self.data[0][2] = value

    @property
    def m03(self) -> np.float32:
        return self.data[0][3]

    @m03.setter
    def m03(self, value) -> None:
        self.data[0][3] = value

    @property
    def m10(self) -> np.float32:
        return self.data[1][0]

    @m10.setter
    def m10(self, value) -> None:
        self.data[1][0] = value

    @property
    def m11(self) -> np.float32:
        return self.data[1][1]

    @m11.setter
    def m11(self, value) -> None:
        self.data[1][1] = value

    @property
    def m12(self) -> np.float32:
        return self.data[1][2]

    @m12.setter
    def m12(self, value) -> None:
        self.data[1][2] = value

    @property
    def m13(self) -> np.float32:
        return self.data[1][3]

    @m13.setter
    def m13(self, value) -> None:
        self.data[1][3] = value

    @property
    def m20(self) -> np.float32:
        return self.data[2][0]

    @m20.setter
    def m20(self, value) -> None:
        self.data[2][0] = value

    @property
    def m21(self) -> np.float32:
        return self.data[2][1]

    @m21.setter
    def m21(self, value) -> None:
        self.data[2][1] = value

    @property
    def m22(self) -> np.float32:
        return self.data[2][2]

    @m22.setter
    def m22(self, value) -> None:
        self.data[2][2] = value

    @property
    def m23(self) -> np.float32:
        return self.data[2][3]

    @m23.setter
    def m23(self, value) -> None:
        self.data[2][3] = value

    @property
    def m30(self) -> np.float32:
        return self.data[3][0]

    @m30.setter
    def m30(self, value) -> None:
        self.data[3][0] = value

    @property
    def m31(self) -> np.float32:
        return self.data[3][1]

    @m31.setter
    def m31(self, value) -> None:
        self.data[3][1] = value

    @property
    def m32(self) -> np.float32:
        return self.data[3][2]

    @m32.setter
    def m32(self, value) -> None:
        self.data[3][2] = value

    @property
    def m33(self) -> np.float32:
        return self.data[3][3]

    @m33.setter
    def m33(self, value) -> None:
        self.data[3][3] = value


class Vector2f:
    def __init__(self, x=None, y=None, dtype=np.int32):
        if x is None or y is None:
            self.vec = np.zeros((1, 2), dtype=dtype)
        else:
            self.vec = np.array([x, y])

    def set(self, new: 'Vector2f') -> None:
        """
        Sets the value of self to the value of new
        Args:
            new (Vector2f): the vector containing the new values.

        Returns:
            None
        """
        self.x = new.x
        self.y = new.y

    @property
    def x(self) -> int:
        return self.vec[0]

    @x.setter
    def x(self, value) -> None:
        self.vec[0] = value

    @property
    def y(self) -> int:
        return self.vec[1]

    @y.setter
    def y(self, value) -> None:
        self.vec[1] = value


class Vector3f:
    def __init__(self, x=None, y=None, z=None, dtype=np.int32):
        if x is None or y is None or z is None:
            self.vec = np.zeros((1, 3), dtype=dtype)
        else:
            self.vec = np.array([x, y, z])

    def set(self, new: 'Vector3f') -> None:
        """
        Sets the value of self to the value of new
        Args:
            new (Vector3f): the vector containing the new values.

        Returns:
            None
        """
        self.x = new.x
        self.y = new.y
        self.z = new.z

    @property
    def x(self) -> int:
        return self.vec[0]

    @x.setter
    def x(self, value) -> None:
        self.vec[0] = value

    @property
    def y(self) -> int:
        return self.vec[1]

    @y.setter
    def y(self, value) -> None:
        self.vec[1] = value

    @property
    def z(self) -> int:
        return self.vec[2]

    @z.setter
    def z(self, value) -> None:
        self.vec[2] = value


class Vector4f:
    def __init__(self, x=None, y=None, z=None, w=None, dtype=np.int32):
        if x is None or y is None or z is None or w is None:
            self.vec = np.zeros((1, 4), dtype=dtype)
        else:
            self.vec = np.array([x, y, z, w])

    def set(self, new: 'Vector4f') -> None:
        """
        Sets the value of self to the value of new
        Args:
            new (Vector4f): the vector containing the new values.

        Returns:
            None
        """
        self.x = new.x
        self.y = new.y
        self.z = new.z
        self.w = new.w

    @property
    def x(self) -> int:
        return self.vec[0]

    @x.setter
    def x(self, value) -> None:
        self.vec[0] = value

    @property
    def y(self) -> int:
        return self.vec[1]

    @y.setter
    def y(self, value) -> None:
        self.vec[1] = value

    @property
    def z(self) -> int:
        return self.vec[2]

    @z.setter
    def z(self, value) -> None:
        self.vec[2] = value

    @property
    def w(self) -> int:
        return self.vec[3]

    @w.setter
    def w(self, value) -> None:
        self.vec[3] = value
