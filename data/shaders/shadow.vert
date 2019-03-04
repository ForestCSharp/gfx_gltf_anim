#version 450

layout( location = 0 ) in vec3 in_pos;

layout( set = 0, binding = 0 ) uniform UniformBuffer {
    mat4 ShadowMVP;
};

void main() {
  gl_Position = ShadowMVP * vec4(in_pos, 1.0);
}