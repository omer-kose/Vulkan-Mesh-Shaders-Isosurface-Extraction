#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_GOOGLE_include_directive : enable

#include "../global_input.h"

// Disgusting solution. Not needed but have to declare to be able to use existing ChunkMetadata data
layout(buffer_reference, scalar) readonly buffer VoxelBuffer
{
	uint8_t voxels[];
};

/*
	Indices of the chunks that have possibility for extracting surface given the isovalue
*/
layout(buffer_reference, scalar) readonly buffer ActiveChunkIndicesBuffer
{
	uint activeChunkIndices[];
};

/*
	Chunk metadata unique to that chunk. This buffer already exist on the GPU side so I am just reusing it
*/
struct ChunkMetadata
{
	// Positional Limits of the Grid
	vec3 lowerCornerPos;
	vec3 upperCornerPos;
	VoxelBuffer voxelBufferDeviceAddress; // not used in this shader
};

layout(buffer_reference, scalar) readonly buffer ChunkMetadataBuffer
{
	ChunkMetadata chunkMetadata[];
};

layout(push_constant, scalar) uniform PushConstants
{
	ChunkMetadataBuffer chunkMetadataBuffer;
	ActiveChunkIndicesBuffer activeChunkIndicesBuffer;
	uint numActiveChunks;
};