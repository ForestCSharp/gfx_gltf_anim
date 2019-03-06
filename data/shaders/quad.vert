#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(constant_id = 0) const float scale = 2.0f;

layout(binding = 0) uniform UniformStruct {
    mat4 view_matrix;
    mat4 proj_matrix;
    mat4 model_matrix;
    float time;
} ubo;

#define MAX_BONE_COUNT 100

layout(binding = 1) uniform SkeletonUniform {
    mat4 bones[MAX_BONE_COUNT];
} skeleton;

layout( set = 0, binding = 2 ) uniform UniformBuffer {
    mat4 ShadowMVP;
};

//FIXME: ensure this is correct shadow bias matrix
const mat4 shadowBiasMatrix = 
mat4( 
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0 
    );

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec4 in_col;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec3 in_norm;
layout(location = 4) in vec4 in_joint_indices;
layout(location = 5) in vec4 in_joint_weights;

layout(location = 0) out vec4 out_pos;
layout(location = 1) out vec4 out_col;
layout(location = 2) out vec2 out_uv;
layout(location = 3) out vec3 out_norm;
layout(location = 4) out vec4 out_shadow_coord;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {

    mat4 skinMatrix = mat4(0.0);

    for (int i=0; i<4; ++i)
    {
        skinMatrix += (in_joint_weights[i] * skeleton.bones[int(in_joint_indices[i])]);
    }

    //Skin matrix is identity if joint weights are all zero
    if ((abs(in_joint_weights[0] - 0.0)) < 0.000001)
    {
        skinMatrix = mat4(1.0);
    }

    //Offset Instances based on gl_InstanceIndex
    mat4 Translation = mat4(0.0);
    Translation[0][0] = 1;
    Translation[1][1] = 1;
    Translation[2][2] = 1;
    Translation[3][3] = 1;
    Translation[3][0] = 1.0 * (gl_InstanceIndex / 10) - 10.0;
    Translation[3][2] = -1.0 * mod(gl_InstanceIndex,10);

    mat4 ModelMatrix = ubo.model_matrix * Translation;
    
    //TODO: handle
    out_pos = vec4(in_pos, 1.0);
    
    //TODO: will need to do this in the tessellation eval shader if tessellated
    //out_pos = (ubo.proj_matrix * ubo.view_matrix * ModelMatrix * skinMatrix * vec4(in_pos, 1.0));
    out_col = in_col;
    out_uv = in_uv;
    out_norm = in_norm;
    out_shadow_coord = shadowBiasMatrix * ShadowMVP * vec4(in_pos, 1.0);
    gl_Position = out_pos;
}
