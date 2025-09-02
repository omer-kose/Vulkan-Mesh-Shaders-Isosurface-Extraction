#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_GOOGLE_include_directive : enable

#include "../global_input.h"

#define BLOCK_SIZE 4

// Disgusting solution. Not needed but have to declare to be able to use existing ChunkMetadata data
layout(buffer_reference, scalar) readonly buffer VoxelBuffer
{
	uint8_t voxels[];
};


struct ChunkDrawData
{
	uint chunkID; // ID of the chunk in the ChunkMetadata array
	// Note: This actually could be computed on the fly: workGroupID % (numGroupsPerChunk). But for ease in debugging, I will keep this explicit.
	uint localWorkgroupID; // Explicitly assign a local work group ID working on that chunk. In other words, this is the id of the block that task shader will work on in the chunk. Range: [0, numGroupsPerChunk - 1]
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

/*
	Blocks that have been rendered last frame
*/
layout(buffer_reference, scalar) buffer ChunkDrawDataBuffer
{
	ChunkDrawData chunkDrawData[];
};

layout(push_constant, scalar) uniform PushConstants
{
	ChunkMetadataBuffer chunkMetadataBuffer;
	ChunkDrawDataBuffer chunkDrawDataBuffer;
	uvec3 chunkSize;
};