#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformStruct {
    mat4 view_matrix;
    mat4 proj_matrix;
    mat4 model_matrix;
    float time;
} ubo;

layout( set = 0, binding = 2 ) uniform UniformBuffer {
    mat4 ShadowMVP;
    vec3 light_dir;
    float shadow_bias;
};

layout(set = 0, binding = 3) uniform sampler2D shadow_sampler;

layout(location = 0) in vec4 in_pos;
layout(location = 1) in vec4 in_col;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec3 in_norm;
layout(location = 4) in vec4 in_shadow_coord;

layout(location = 0) out vec4 target0;

float textureProj(vec4 shadowCoord, vec2 off, float bias)
{
	float shadow = 1.0;
	if ( shadowCoord.z > -1.0 && shadowCoord.z < 1.0 ) 
	{
		float dist = texture( shadow_sampler, shadowCoord.st + off ).r - bias;
		if ( shadowCoord.w > 0.0 && dist > shadowCoord.z ) 
		{
			shadow = 0.0;
		}
	}
	return shadow;
}

float filterPCF(vec4 sc, float bias)
{
	ivec2 texDim = textureSize(shadow_sampler, 0);
	float scale = 1.5;
	float dx = scale * 1.0 / float(texDim.x);
	float dy = scale * 1.0 / float(texDim.y);

	float shadowFactor = 0.0;
	int count = 0;
	int range = 1;
	
	for (int x = -range; x <= range; x++)
	{
		for (int y = -range; y <= range; y++)
		{
			shadowFactor += textureProj(sc, vec2(dx*x, dy*y), bias);
			count++;
		}
	
	}
	return shadowFactor / count;
}

void main() {

    int int_time = int(ubo.time / 8.0); //Divide to slow rate of change
    int remainder = int_time % 2; //Gives you n options in switch

    //Basic Directional Diffuse Lighting
    vec4 diffuse = vec4(0.8, 0.7, 0.7, 1.0);
    float shininess = 1.0;

    vec3 l = normalize(light_dir);
    vec3 n = normalize(in_norm);
    float intensity = max(dot(n,l), 0.0);

    float bias_factor = shadow_bias;
    float bias = max(bias_factor * (1.0 - dot(n, l)), bias_factor / 10.0);
    bool enablePCF = true;
    //FIXME: turn shadow mapping back on
    //intensity = enablePCF ? filterPCF(in_shadow_coord / in_shadow_coord.w, bias) : textureProj(in_shadow_coord / in_shadow_coord.w, vec2(0.0), bias);

    const float ambient = 0.05;
    intensity += ambient;

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

    col = vec4(in_col.xyz * intensity, 1.0);

    target0 = col;
}
