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

layout(triangles) in;

layout(binding = 0) uniform UniformStruct {
    mat4 view_matrix;
    mat4 proj_matrix;
    mat4 model_matrix;
    float time;
    float pn_triangles_strength;
    float tess_level;
} ubo;

layout( set = 0, binding = 2 ) uniform UniformBuffer {
    mat4 ShadowMVP;
    vec3 light_dir;
};

layout(location = 0) in vec4 in_pos[];
layout(location = 1) in vec4 in_col[];
layout(location = 2) in vec2 in_uv[];
layout(location = 3) in vec3 in_norm[];
layout(location = 4) in vec4 in_shadow_coord[];
layout(location = 5) in PnPatch in_patch[];

layout(location = 0) out vec4 out_pos;
layout(location = 1) out vec4 out_col;
layout(location = 2) out vec2 out_uv;
layout(location = 3) out vec3 out_norm;
layout(location = 4) out vec4 out_shadow_coord;

void main()
{
    vec3 uvw = gl_TessCoord;
    vec3 uvwSquared = uvw * uvw;
    vec3 uvwCubed = uvwSquared * uvw;

    // extract control points
    vec3 b210 = vec3(in_patch[0].b210, in_patch[1].b210, in_patch[2].b210);
    vec3 b120 = vec3(in_patch[0].b120, in_patch[1].b120, in_patch[2].b120);
    vec3 b021 = vec3(in_patch[0].b021, in_patch[1].b021, in_patch[2].b021);
    vec3 b012 = vec3(in_patch[0].b012, in_patch[1].b012, in_patch[2].b012);
    vec3 b102 = vec3(in_patch[0].b102, in_patch[1].b102, in_patch[2].b102);
    vec3 b201 = vec3(in_patch[0].b201, in_patch[1].b201, in_patch[2].b201);
    vec3 b111 = vec3(in_patch[0].b111, in_patch[1].b111, in_patch[2].b111);

    // extract control normals
    vec3 n110 = normalize(vec3(in_patch[0].n110, in_patch[1].n110, in_patch[2].n110));
    vec3 n011 = normalize(vec3(in_patch[0].n011, in_patch[1].n011, in_patch[2].n011));
    vec3 n101 = normalize(vec3(in_patch[0].n101, in_patch[1].n101, in_patch[2].n101));

    out_uv = gl_TessCoord.x * in_uv[0] + gl_TessCoord.y * in_uv[1] + gl_TessCoord.z * in_uv[2];
    out_col = gl_TessCoord.x * in_col[0] + gl_TessCoord.y * in_col[1] + gl_TessCoord.z * in_col[2];
    //out_shadow_coord = gl_TessCoord.x * in_shadow_coord[0] + gl_TessCoord.y * in_shadow_coord[1] + gl_TessCoord.z * in_shadow_coord[2];

    //TODO: make this tweakable
    float tessAlpha = ubo.pn_triangles_strength;
    // normal
    // Barycentric normal
    vec3 barNormal = gl_TessCoord[2] * in_norm[0] + gl_TessCoord[0] * in_norm[1] + gl_TessCoord[1] * in_norm[2];
    vec3 pnNormal = in_norm[0] * uvwSquared[2] + in_norm[1] * uvwSquared[0] + in_norm[2] * uvwSquared[1] + n110 * uvw[2] * uvw[0] +
                    n011 * uvw[0] * uvw[1] + n101 * uvw[2] * uvw[1];
    out_norm = tessAlpha * pnNormal + (1.0 - tessAlpha) * barNormal;
    
    //NOTE: curved tries with un-tessellated normals
    //out_norm = gl_TessCoord.x * in_norm[0] + gl_TessCoord.y * in_norm[1] + gl_TessCoord.z * in_norm[2];

    // compute interpolated pos
    vec3 barPos = gl_TessCoord[2] * in_pos[0].xyz + gl_TessCoord[0] * in_pos[1].xyz + gl_TessCoord[1] * in_pos[2].xyz;

    // save some computations
    uvwSquared *= 3.0;

    // compute PN position
    vec3 pnPos = in_pos[0].xyz * uvwCubed[2] + in_pos[1].xyz * uvwCubed[0] + in_pos[2].xyz * uvwCubed[1] +
                b210 * uvwSquared[2] * uvw[0] + b120 * uvwSquared[0] * uvw[2] + b201 * uvwSquared[2] * uvw[1] + b021 * uvwSquared[0] * uvw[1] +
                b102 * uvwSquared[1] * uvw[2] + b012 * uvwSquared[1] * uvw[0] + b111 * 6.0 * uvw[0] * uvw[1] * uvw[2];

    //FIXME: gpu skinning
    // final position and normal
    vec3 finalPos = (1.0 - tessAlpha) * barPos + tessAlpha * pnPos;
    gl_Position = out_pos = ubo.proj_matrix * ubo.view_matrix * ubo.model_matrix * vec4(finalPos, 1.0);


    const mat4 shadowBiasMatrix = 
    mat4( 
        0.5, 0.0, 0.0, 0.0,
        0.0, 0.5, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.5, 0.5, 0.0, 1.0 
    );

    out_shadow_coord = shadowBiasMatrix * ShadowMVP * vec4(finalPos, 1.0);
}