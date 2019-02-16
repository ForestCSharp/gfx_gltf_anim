#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformStruct {
    mat4 view_matrix;
    mat4 proj_matrix;
    mat4 model_matrix;
    float time;
} ubo;

layout(location = 0) in vec4 in_pos;
layout(location = 1) in vec4 in_col;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec3 in_norm;

layout(location = 0) out vec4 target0;

void main() {

    int int_time = int(ubo.time / 2.0); //Divide to slow rate of change
    int remainder = int_time % 2; //Gives you 4 options in switch

    //Basic Directional Diffuse Lighting
    vec4 diffuse = vec4(0.8, 0.7, 0.7, 1.0);
    float ambient = 0.2;
    float shininess = 1.0;

    vec3 l = normalize(vec3(-1, -1, 0));
    vec3 n = normalize(in_norm);
    float intensity = max(dot(n,l), 0.0);

    vec4 lighting_color = max(intensity * diffuse, vec4(diffuse.xyz * ambient, 1.0));

    vec4 col;
    switch (remainder) 
    {
        case 0:
                col = lighting_color;
                break;
        case 1:
                col = vec4(in_norm, 1.0);
                break;
        default:
                col = in_col;
    }

    target0 = col;
}
