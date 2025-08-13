#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_8bit_storage : require

layout(set = 0, binding = 0, scalar) uniform SceneData
{
	mat4 view;
	mat4 proj;
	mat4 viewproj;
	vec4 ambientColor;
	vec4 sunlightDirection; //w for sun power
	vec4 sunlightColor;
} sceneData;

#define BLOCK_SIZE 4  // block size that each group processes (e.g., 4x4x4)

// Derived Constants
#define BLOCK_PLUS_1 (BLOCK_SIZE + 1)  // N+1 samples needed for N cubes
#define BLOCK_VOLUME (BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE)

// MC Settings that are sent to the gpu with the Push Constant (if using Chunks, MCSettings is the common parameter for all the chunks (MC only sees voxelBuffer assigned to it so with or without chunks the same pass is used))
struct MCSettings
{
	uvec3 gridSize; // Either determined by the input data or the user if a custom SDF is used (such as a noise function)
	uvec3 shellSize; // For chunks a shell with +2 on right-bottom-front boundaries for correct computation. For a non-chunked volume gridSize==shellSize. This is only used for fetching the data correctly with voxelValue()
	float isoValue;
};

layout(buffer_reference, scalar) readonly buffer VoxelBuffer
{
	float voxels[];
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

layout(buffer_reference, scalar) readonly buffer ChunkMetadataBuffer
{
	ChunkMetadata chunkMetadata[];
};

layout(buffer_reference, scalar) buffer ChunkDrawDataBuffer
{
	ChunkDrawData chunkDrawData[];
};

/*
	Indices of the chunks that have possibility for extracting surface given the isovalue
*/
layout(buffer_reference, scalar) readonly buffer ActiveChunkIndicesBuffer
{
	uint activeChunkIndices[];
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
	MCSettings mcSettings;
	float zNear;
	uint depthPyramidWidth;
	uint depthPyramidHeight;
	ChunkMetadataBuffer chunkMetadataBuffer;
	ChunkDrawDataBuffer chunkDrawDataBuffer;
	ActiveChunkIndicesBuffer activeChunkIndicesBuffer;
	uint numActiveChunks;
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
	uint chunkID;
};

/*
	Given 3D voxel idx in indices (within bounds, [0, gridSize-1]), fetches the value from the voxel buffer.

	Note that, voxel values actually represent values at the corners not per-voxel center value in the context of Marching Cubes. In Marching Cubes, each corner actually has a SDF value. So, per voxel there are 8 values

	For example, voxels[0] is the value of voxel 0's 0'th corner's value. voxels[1] lies at one right (I store the values in z-y-x order) and so on. In other words, a voxel i's value actually means
	the values lies in its first corner. 

	In this indirect version, the base address to the buffer will be fetched from ChunkMetadata dynamically depending on the chunkID. So, the thread will provide the chunkID.
*/
float voxelValue(uint chunkID, uvec3 idx)
{
	return chunkMetadataBuffer.chunkMetadata[chunkID].voxelBufferDeviceAddress.voxels[idx.x + mcSettings.shellSize.x * (idx.y + mcSettings.shellSize.y * idx.z)];
}

bool projectBox(vec3 bmin, vec3 bmax, float znear, mat4 viewProjection, out vec4 aabb)
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

	if(min(P0.w, min(P1.w, min(P2.w, min(P3.w, min(P4.w, min(P5.w, min(P6.w, P7.w))))))) < znear) return false;

	aabb.xy = min(
		P0.xy / P0.w, min(P1.xy / P1.w, min(P2.xy / P2.w, min(P3.xy / P3.w,
		min(P4.xy / P4.w, min(P5.xy / P5.w, min(P6.xy / P6.w, P7.xy / P7.w)))))));
	aabb.zw = max(
		P0.xy / P0.w, max(P1.xy / P1.w, max(P2.xy / P2.w, max(P3.xy / P3.w,
		max(P4.xy / P4.w, max(P5.xy / P5.w, max(P6.xy / P6.w, P7.xy / P7.w)))))));

	// clip space -> uv space
	aabb = aabb * vec4(0.5f, 0.5f, 0.5f, 0.5f) + vec4(0.5f);

	return true;
}

