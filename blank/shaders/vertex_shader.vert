#version 130

in vec3 position;
in vec2 texture_coords;

uniform mat4 transform;

out vec2 out_texture_coords;

void main(void){
    vec4 world_pos = transform * vec4(position, 1.0);
    gl_Position = world_pos;

    out_texture_coords = texture_coords;
}