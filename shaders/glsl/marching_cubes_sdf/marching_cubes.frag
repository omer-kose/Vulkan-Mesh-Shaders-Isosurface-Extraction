#version 450

#extension GL_GOOGLE_include_directive : require
#include "input_structures.h"

layout (location = 0) out vec4 outFragColor;

layout(location = 0) in MeshIn
{
	vec3 normal;
} meshIn;

void main()
{
	vec3 normal = meshIn.normal;
	float lightValue = max(dot(normal, sceneData.sunlightDirection.xyz), 0.1f);

	vec3 color = vec3(0.5f);
	vec3 ambient = color *  sceneData.ambientColor.xyz;

	outFragColor = vec4(color * lightValue * sceneData.sunlightColor.w + ambient, 1.0f);
}