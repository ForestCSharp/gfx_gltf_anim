#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(constant_id = 0) const float scale = 2.0f;

layout(binding = 0) uniform CameraUniform {
    mat4 view_matrix;
    mat4 proj_matrix;
    mat4 model_matrix;
    float time;
} cam;

#define MAX_BONE_COUNT 100

layout(binding = 1) uniform SkeletonUniform {
    mat4 bones[MAX_BONE_COUNT];
} skeleton;
//End Skeleton Uniform Data

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec4 a_col;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in vec3 a_norm;
layout(location = 4) in vec4 a_joint_indices;
layout(location = 5) in vec4 a_joint_weights;

layout(location = 0) out vec4 v_pos;
layout(location = 1) out vec4 v_col;
layout(location = 2) out vec2 v_uv;
layout(location = 3) out vec3 v_norm;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {

    mat4 skinMatrix = mat4(0.0);

    for (int i=0; i<4; ++i)
    {
        skinMatrix += (a_joint_weights[i] * skeleton.bones[int(a_joint_indices[i])]);
    }

    //Skin matrix is identity if joint weights are all zero
    if ((abs(a_joint_weights[0] - 0.0)) < 0.000001)
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

    mat4 ModelMatrix = cam.model_matrix * Translation;

    vec3 color = gl_VertexIndex % 3 == 0 ? vec3(1,0,0) : (gl_VertexIndex % 3 == 1 ? vec3(0,1,0) : vec3(0,0,1));

    v_pos = (cam.proj_matrix * cam.view_matrix * ModelMatrix * skinMatrix * vec4(a_pos, 1.0));
    v_col = vec4(color, 1);
    v_uv = a_uv;
    v_norm = a_norm;
    gl_Position = v_pos;
}
