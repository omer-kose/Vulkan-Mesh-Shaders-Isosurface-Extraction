#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_8bit_storage : require

#include "input_structures.h"

layout (location = 0) out vec4 outFragColor;

layout(location = 0) in MeshIn
{
	vec3 normal;
    flat uint colorIndex;
} meshIn;

void main()
{
	vec3 normal = meshIn.normal;
	uint colorIndex = meshIn.colorIndex;
	float lightValue = max(dot(normal, -sceneData.sunlightDirection.xyz), 0.1f);

	vec3 color = vec3(uint(palette[colorIndex].color[0]) / 255.0, uint(palette[colorIndex].color[1]) / 255.0, uint(palette[colorIndex].color[2]) / 255.0);
	vec3 ambient = color *  sceneData.ambientColor.xyz;

	outFragColor = vec4(color * lightValue * sceneData.sunlightColor.w + ambient, 1.0f);
}