import os

from blank.shaders.shaders import StaticShader
from vulcan.engine import Display, Loader, Renderer
from vulcan.engine.objects import Entity, TransformComponent, MaterialComponent
from vulcan.engine.utils import Vector2f


def main():
    """
    start point
    """
    Display.init()

    shader = StaticShader()
    renderer = Renderer(shader=shader)

    entity = Entity('e')
    entity.attach(TransformComponent(Vector2f(0.7, 0)))
    entity.attach(MaterialComponent(Loader.load_material(1, 1, os.path.abspath('res/img.png')), 1, 1))

    while not Display.window_closed():
        # render stuff
        renderer.prepare()
        shader.start()
        renderer.render(material=entity.get_component(MaterialComponent).material,
                        transformation_matrix=entity.get_component(TransformComponent).get_transformation_matrix())
        shader.stop()

        Display.swap_buffers()
        Display.poll_events()

    shader.clean_up()
    Loader.clean_up()
    Display.terminate()


if __name__ == "__main__":
    main()
