#version 450
#extension GL_ARB_separate_shader_objects : enable

// PN patch data
struct PnPatch
{
  float b210;
  float b120;
  float b021;
  float b012;
  float b102;
  float b201;
  float b111;
  float n110;
  float n011;
  float n101;
};

layout(vertices = 3) out;

layout(binding = 0) uniform UniformStruct {
    mat4 view_matrix;
    mat4 proj_matrix;
    mat4 model_matrix;
    float time;
    float pn_triangles_strength;
    float tess_level;
} ubo;

layout(location = 0) in vec4 in_pos[];
layout(location = 1) in vec4 in_col[];
layout(location = 2) in vec2 in_uv[];
layout(location = 3) in vec3 in_norm[];

layout(location = 0) out vec4 out_pos[];
layout(location = 1) out vec4 out_col[];
layout(location = 2) out vec2 out_uv[];
layout(location = 3) out vec3 out_norm[];
layout(location = 4) out PnPatch out_patch[3];

float wij(int i, int j)
{
    return dot(in_pos[j].xyz - in_pos[i].xyz, in_norm[i]);
}

float vij(int i, int j)
{
    vec3 Pj_minus_Pi = in_pos[j].xyz - in_pos[i].xyz;
    vec3 Ni_plus_Nj = in_norm[i] + in_norm[j];
    return 2.0 * dot(Pj_minus_Pi, Ni_plus_Nj) / dot(Pj_minus_Pi, Pj_minus_Pi);
}

void main()
{
    out_pos[gl_InvocationID] = in_pos[gl_InvocationID];
    out_col[gl_InvocationID] = in_col[gl_InvocationID];
    out_norm[gl_InvocationID] = in_norm[gl_InvocationID];
    out_uv[gl_InvocationID] = in_uv[gl_InvocationID];

    // set base
    float P0 = in_pos[0][gl_InvocationID];
    float P1 = in_pos[1][gl_InvocationID];
    float P2 = in_pos[2][gl_InvocationID];
    float N0 = in_norm[0][gl_InvocationID];
    float N1 = in_norm[1][gl_InvocationID];
    float N2 = in_norm[2][gl_InvocationID];

    // compute control points
    out_patch[gl_InvocationID].b210 = (2.0 * P0 + P1 - wij(0, 1) * N0) / 3.0;
    out_patch[gl_InvocationID].b120 = (2.0 * P1 + P0 - wij(1, 0) * N1) / 3.0;
    out_patch[gl_InvocationID].b021 = (2.0 * P1 + P2 - wij(1, 2) * N1) / 3.0;
    out_patch[gl_InvocationID].b012 = (2.0 * P2 + P1 - wij(2, 1) * N2) / 3.0;
    out_patch[gl_InvocationID].b102 = (2.0 * P2 + P0 - wij(2, 0) * N2) / 3.0;
    out_patch[gl_InvocationID].b201 = (2.0 * P0 + P2 - wij(0, 2) * N0) / 3.0;
    float E = (out_patch[gl_InvocationID].b210 + out_patch[gl_InvocationID].b120 + out_patch[gl_InvocationID].b021 + out_patch[gl_InvocationID].b012 +
                out_patch[gl_InvocationID].b102 + out_patch[gl_InvocationID].b201) /
                6.0;
    float V = (P0 + P1 + P2) / 3.0;
    out_patch[gl_InvocationID].b111 = E + (E - V) * 0.5;
    out_patch[gl_InvocationID].n110 = N0 + N1 - vij(0, 1) * (P1 - P0);
    out_patch[gl_InvocationID].n011 = N1 + N2 - vij(1, 2) * (P2 - P1);
    out_patch[gl_InvocationID].n101 = N2 + N0 - vij(2, 0) * (P0 - P2);

    // set tess levels
    // TODO: variable for tess level
    gl_TessLevelOuter[gl_InvocationID] = ubo.tess_level;
    gl_TessLevelInner[0] = ubo.tess_level;
}