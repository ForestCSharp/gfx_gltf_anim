#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(triangles) in;

layout(binding = 0) uniform UniformStruct {
    mat4 view_matrix;
    mat4 proj_matrix;
    mat4 model_matrix;
    float time;
} ubo;

layout(location = 0) in vec4 in_pos[];
layout(location = 1) in vec4 in_col[];
layout(location = 2) in vec2 in_uv[];
layout(location = 3) in vec3 in_norm[];

layout(location = 0) out vec4 out_pos;
layout(location = 1) out vec4 out_col;
layout(location = 2) out vec2 out_uv;
layout(location = 3) out vec3 out_norm;

void main()
{
    //FIXME: gpu skinning
    gl_Position = (gl_TessCoord.x * in_pos[0]) + (gl_TessCoord.y * in_pos[1]) + (gl_TessCoord.z * in_pos[2]);
    gl_Position = ubo.proj_matrix * ubo.view_matrix * ubo.model_matrix * gl_Position;
    out_pos = gl_Position;
    
    out_col = gl_TessCoord.x * in_col[0] + gl_TessCoord.y * in_col[1] + gl_TessCoord.z * in_col[2];
    out_norm = gl_TessCoord.x * in_norm[0] + gl_TessCoord.y * in_norm[1] + gl_TessCoord.z * in_norm[2];
    out_uv = gl_TessCoord.x * in_uv[0] + gl_TessCoord.y * in_uv[1] + gl_TessCoord.z * in_uv[2];
}