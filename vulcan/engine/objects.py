from abc import ABCMeta, abstractmethod
from typing import Type, Any, List

from vulcan.engine.utils import Vector2f, create_transformation_matrix, Vector3f, Matrix4f, Material


class Entity:
    def __init__(self, entity_id):
        # type: (str) -> None
        self.id = entity_id
        self._components = list()  # type: List['Component']

    def add_component(self, component):
        # type: (Component) -> None
        """
        Appends a component to the list of components.
        Args:
            component (Component): The component to be added.

        Returns:
            None
        """
        self._components.append(component)

    def remove_component(self, cls):
        # type: (Type[Component]) -> None
        """
        Removes a component that is the instance of cls.
        Args:
            cls: The class of the component to be removed

        Returns:
            None
        """
        for c in self._components:
            if isinstance(c, cls):
                self._components.remove(c)

    def get_component(self, cls):
        # type: (Type[Component]) -> Any
        """
        returns the instance of cls present in the list of components.
        Args:
            cls: The class of the component to be obtained.

        Returns:
            (Component): The instance found in the list of components
        """

        for c in self._components:
            if isinstance(c, cls):
                return c


class Component(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def update(self):
        pass


class TransformComponent(Component):
    def __init__(self, position, rotation=Vector3f(0, 0, 0), scale=Vector3f(1, 1, 1)):
        # type: (Vector2f, Vector3f, Vector3f) -> None

        super(TransformComponent, self).__init__()
        self._position = position
        self._rotation = rotation
        self._scale = scale

    def get_transformation_matrix(self):
        # type: () -> Matrix4f
        """
        Returns the transformation matrix created from the translation, rotation and scale.
        Returns:
            Matrix4f: The transformation matrix
        """
        return create_transformation_matrix(translation=Vector3f(self._position.x, self._position.y, 0),
                                            rot=self._rotation, scale=self._scale)

    def translate(self, new_position):
        # type: (Vector2f) -> None
        """
        Changes the position of the entity.
        Args:
            new_position (Vector2f): The new value of position

        Returns:
            None
        """
        self._position.set(new_position)

    def rotate(self, new_rotation):
        # type: (Vector3f) -> None
        """
        Changes the rotation of the entity.
        Args:
            new_rotation (Vector3f): The new value of rotation

        Returns:
            None
        """
        self._rotation.set(new_rotation)

    def scale(self, new_scale):
        # type: (Vector3f) -> None
        """
        Changes the scale of the entity
        Args:
            new_scale (Vector3f): The new value of scale

        Returns:
            None
        """
        self._scale.set(new_scale)

    def update(self):
        pass


class MaterialComponent(Component):
    def __init__(self, material, sx, sy):
        # type: (Material, float, float) -> None

        super(MaterialComponent, self).__init__()
        self.material = material
        self.sx = sx
        self.sy = sy

    def update(self):
        pass
