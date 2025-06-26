#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference : require

struct ChunkVisInformation
{
	vec3 lowerCornerPos;
	vec3 upperCornerPos;
	float minIsoValue;
	float maxIsoValue;
};

layout(buffer_reference, scalar) readonly buffer ChunkBuffer
{
	ChunkVisInformation chunks[];
};

layout(set = 0, binding = 0, scalar) uniform SceneData
{
	mat4 view;
	mat4 proj;
	mat4 viewproj;
	vec4 ambientColor;
	vec4 sunlightDirection; //w for sun power
	vec4 sunlightColor;
} sceneData;

layout(push_constant, scalar) uniform PushConstants
{
	ChunkBuffer chunkBuffer;
	float inputIsoValue;
} pushConstants;