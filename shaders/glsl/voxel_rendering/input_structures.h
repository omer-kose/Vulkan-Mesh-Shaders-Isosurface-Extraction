#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_GOOGLE_include_directive : enable

#include "../global_input.h"

#define BLOCK_SIZE 4  // block size that each group processes (e.g., 4x4x4)

// Derived Constants
#define BLOCK_PLUS_1 (BLOCK_SIZE + 1)  // N+1 samples needed for N cubes
#define BLOCK_VOLUME (BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE)

layout(buffer_reference, scalar) readonly buffer VoxelBuffer
{
	uint8_t voxels[]; // Contains color data. If 0 that means Voxel is not solid (not occupied).
};

/*
	Chunk metadata unique to that chunk. Common values are stored in MCSettings above
*/
struct ChunkMetadata
{
	// Positional Limits of the Grid
	vec3 lowerCornerPos;
	vec3 upperCornerPos;
	VoxelBuffer voxelBufferDeviceAddress; // base address of the voxels of the chunk in the Voxel Buffer
};

/*
	Necessary information for the task and mesh shaders to be able to fetch required chunk data for dispatch. This will be filled by the compute shader for each task shader dispatch.

	Each task shader invocation has a unique ChunkDrawData entry.
*/
struct ChunkDrawData
{
	uint chunkID; // ID of the chunk in the ChunkMetadata array
	// Note: This actually could be computed on the fly: workGroupID % (numGroupsPerChunk). But for ease in debugging, I will keep this explicit.
	uint localWorkgroupID; // Explicitly assign a local work group ID working on that chunk. In other words, this is the id of the block that task shader will work on in the chunk. Range: [0, numGroupsPerChunk - 1]
};

struct VoxelColor
{
	uint8_t color[4]; // also compatible with vox files
};

layout(set = 1, binding = 0, scalar) uniform Palette
{
	VoxelColor palette[256];
};

layout(buffer_reference, scalar) readonly buffer ChunkMetadataBuffer
{
	ChunkMetadata chunkMetadata[];
};

layout(buffer_reference, scalar) buffer ChunkDrawDataBuffer
{
	ChunkDrawData chunkDrawData[];
};


/*
	Final number of chunks that are guaranteed to be processed by task shaders.

	2 dummy values for y and z dispatches which will be always zero. Putting them there to match with the VkDrawMeshTasksIndirectCommandEXT struct as this will be also used as the indirect buffer
*/
layout(buffer_reference, scalar) buffer DrawChunkCountBuffer
{
	uint drawChunkCount;
	uint _dummy1;
	uint _dummy2;
};

layout(push_constant, scalar) uniform PushConstants
{
	uvec3 chunkSize;
	uvec3 shellSize; // For chunks a shell with +2 on right-bottom-front boundaries for correct computation. For voxel rendering, only +1 is enough to check neighbor occupation but I use the same chuking strategy for both MC and Voxel rendering
	vec3 voxelSize; // Size of a singular voxel. All the voxels are uniformly shaped
	uint numChunks;
	float zNear;
	uint depthPyramidWidth;
	uint depthPyramidHeight;
	ChunkMetadataBuffer chunkMetadataBuffer;
	ChunkDrawDataBuffer chunkDrawDataBuffer;
	DrawChunkCountBuffer drawChunkCountBuffer;
};

// Task shader to mesh shader 
struct MeshletData
{
	uint meshletID;
};

struct TaskPayload
{
	MeshletData meshlets[64];
	uint colorIndex[64]; // index to the palette which is equal to voxel value
	uint chunkID;
};

/*
	Fetches the voxel value that stores color value per voxel
*/
uint voxelValue(uint chunkID, uvec3 idx)
{
	return uint(chunkMetadataBuffer.chunkMetadata[chunkID].voxelBufferDeviceAddress.voxels[idx.x + shellSize.x * (idx.y + shellSize.y * idx.z)]);
}

bool projectBox(vec3 bmin, vec3 bmax, float znear, mat4 viewProjection, out vec4 aabb, out float nearestDepth)
{
	vec4 SX = viewProjection * vec4(bmax.x - bmin.x, 0.0, 0.0, 0.0);
	vec4 SY = viewProjection * vec4(0.0, bmax.y - bmin.y, 0.0, 0.0);
	vec4 SZ = viewProjection * vec4(0.0, 0.0, bmax.z - bmin.z, 0.0);

	vec4 P0 = viewProjection * vec4(bmin.x, bmin.y, bmin.z, 1.0);
	vec4 P1 = P0 + SZ;
	vec4 P2 = P0 + SY;
	vec4 P3 = P2 + SZ;
	vec4 P4 = P0 + SX;
	vec4 P5 = P4 + SZ;
	vec4 P6 = P4 + SY;
	vec4 P7 = P6 + SZ;

	/*
		Near-Plane Rejection
		There are two possible approaches here:
		1- Use min to cull the box totally, if any of the corners are behind the camera
		2- Use max to cull the box totally, if all of the corners are behind the camera

		Here, I prefer culling if the box is entirely behind the camera 
	*/
	if(max(P0.w, max(P1.w, max(P2.w, max(P3.w, max(P4.w, max(P5.w, max(P6.w, P7.w))))))) < znear) return false;

	aabb.xy = min(
		P0.xy / P0.w, min(P1.xy / P1.w, min(P2.xy / P2.w, min(P3.xy / P3.w,
			min(P4.xy / P4.w, min(P5.xy / P5.w, min(P6.xy / P6.w, P7.xy / P7.w)))))));
	aabb.zw = max(
		P0.xy / P0.w, max(P1.xy / P1.w, max(P2.xy / P2.w, max(P3.xy / P3.w,
			max(P4.xy / P4.w, max(P5.xy / P5.w, max(P6.xy / P6.w, P7.xy / P7.w)))))));

	// clip space -> uv space
	aabb = aabb * vec4(0.5f, 0.5f, 0.5f, 0.5f) + vec4(0.5f);

	// Avoid divide-by-zero by guarding with a small epsilon
	const float EPS = 1e-9;

	float d0 = (abs(P0.w) > EPS) ? (P0.z / P0.w) : -1e9;
	float d1 = (abs(P1.w) > EPS) ? (P1.z / P1.w) : -1e9;
	float d2 = (abs(P2.w) > EPS) ? (P2.z / P2.w) : -1e9;
	float d3 = (abs(P3.w) > EPS) ? (P3.z / P3.w) : -1e9;
	float d4 = (abs(P4.w) > EPS) ? (P4.z / P4.w) : -1e9;
	float d5 = (abs(P5.w) > EPS) ? (P5.z / P5.w) : -1e9;
	float d6 = (abs(P6.w) > EPS) ? (P6.z / P6.w) : -1e9;
	float d7 = (abs(P7.w) > EPS) ? (P7.z / P7.w) : -1e9;

	// For the reversed-depth convention the closer point has the LARGER value,
	// so take the maximum across corners for a conservative 'nearest' value.
	nearestDepth = max(max(max(d0, d1), max(d2, d3)), max(max(d4, d5), max(d6, d7)));

	return true;
}


