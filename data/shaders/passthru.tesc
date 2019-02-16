#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(vertices = 3) out;

layout(location = 0) in vec4 in_pos[];
layout(location = 1) in vec4 in_col[];
layout(location = 2) in vec2 in_uv[];
layout(location = 3) in vec3 in_norm[];

layout(location = 0) out vec4 out_pos[];
layout(location = 1) out vec4 out_col[];
layout(location = 2) out vec2 out_uv[];
layout(location = 3) out vec3 out_norm[];

void main()
{
  if (gl_InvocationID == 0)
  {
    gl_TessLevelInner[0] = 2.0;
    gl_TessLevelOuter[0] = 2.0;
    gl_TessLevelOuter[1] = 2.0;
    gl_TessLevelOuter[2] = 2.0;
  }

  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
  
  out_pos[gl_InvocationID] = in_pos[gl_InvocationID];
  out_col[gl_InvocationID] = in_col[gl_InvocationID];
  out_norm[gl_InvocationID] = in_norm[gl_InvocationID];
  out_uv[gl_InvocationID] = in_uv[gl_InvocationID];
}