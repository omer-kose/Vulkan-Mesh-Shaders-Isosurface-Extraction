#extension GL_EXT_scalar_block_layout : require

/*
	This file includes global input data needed by all the shaders such as scene uniform buffer.
*/

layout(set = 0, binding = 0, scalar) uniform SceneData
{
    mat4 view;
    mat4 proj;
    mat4 viewproj;
    mat4 prevViewProj; // Previous frame's view projection. Might be useful when temporal reprojection is needed
    vec4 ambientColor;
    vec4 sunlightDirection; // w for sun power
    vec4 sunlightColor;
    vec3 cameraPos;
} sceneData;