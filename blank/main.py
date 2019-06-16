import os

from shaders.shaders import StaticShader
from vulcan.engine import Display, Loader, Renderer
from vulcan.engine.objects import Entity, TransformComponent, MaterialComponent
from vulcan.engine.utils import Vector2f


def main():
    Display.init()

    shader = StaticShader()
    renderer = Renderer()

    entity = Entity('e')
    entity.add_component(TransformComponent(Vector2f(40, 40)))
    entity.add_component(MaterialComponent(Loader.load_material(1, 1, os.path.expanduser('../res/img.png')), 1, 1))

    while not Display.window_closed():
        # render stuff
        renderer.prepare()
        shader.start()
        renderer.render(material=entity.get_component(MaterialComponent).material)
                        # transformation_matrix=entity.get_component(TransformComponent).get_transformation_matrix())
        shader.stop()

        Display.swap_buffers()
        Display.poll_events()

    shader.clean_up()
    Loader.clean_up()
    Display.terminate()


if __name__ == "__main__":
    main()
