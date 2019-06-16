#version 130

//in vec3 out_texture_coords;
in vec2 out_texture_coords;

out vec4 out_color;

uniform sampler2D texture_sampler;

void main(void){
    out_color = texture(texture_sampler, out_texture_coords);
//    out_color = vec4(out_texture_coords, 1);
}