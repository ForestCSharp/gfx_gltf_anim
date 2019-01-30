#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform CameraUniform {
    mat4 view_matrix;
    mat4 proj_matrix;
    mat4 model_matrix;
    float time;
} cam;

layout(location = 0) in vec4 v_pos;
layout(location = 1) in vec4 v_col;
layout(location = 2) in vec2 v_uv;
layout(location = 3) in vec3 v_norm;

layout(location = 0) out vec4 target0;

void main() {

    int int_time = int(cam.time / 2.0); //Divide to slow rate of change
    int remainder = int_time % 4; //Gives you 4 options in switch

    vec4 col;

    switch (remainder) 
    {
        case 0:
                col = v_col;
                break;
        case 1:
                col = vec4(v_norm, 1.0);
                break;
        case 2:
                col = vec4(v_uv, 0.0, 1.0);
                break;
        case 3:
                col = vec4(v_pos.xyz, 1.0);
                break;
        default:
                col = v_col;
    }

    //Basic Directional Lighting
    vec4 diffuse = vec4(0.5, 0.0, 0.0, 1.0);
    vec4 specular = vec4(0.0, 0.0, 0.0, 1.0);
    float ambient = 0.1;
    float shininess = 1.0;

    vec3 l_dir = normalize(vec3(-1, -1, 0));
    vec3 n = normalize(v_norm);
    vec3 e = normalize(-(cam.view_matrix * cam.model_matrix * v_pos).xyz);

    float intensity = max(dot(n,l_dir), 0.0);

    vec4 spec = vec4(0.0);
 
    if (intensity > 0.0) {
        // compute the half vector
        vec3 h = normalize(l_dir + e);  
        // compute the specular term into spec
        float intSpec = max(dot(h,n), 0.0);
        spec = specular * pow(intSpec,shininess);
    }
    
    target0 = max(intensity *  diffuse + spec, vec4(diffuse.xyz * ambient, 1.0));
}
