#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference : require

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
	float isoValue;
};

layout(buffer_reference, scalar) readonly buffer VoxelBuffer
{
	float voxels[];
};

layout(push_constant, scalar) uniform PushConstants
{
	MCSettings mcSettings;
	VoxelBuffer voxelBuffer;
	// Positional Limits of the Grid
	vec3 lowerCornerPos;
	vec3 upperCornerPos;
} pushConstants;

// Task shader to mesh shader 
struct MeshletData
{
	uint meshletID;
};

struct TaskPayload
{
	MeshletData meshlets[64];
};

/*
	Given 3D voxel idx in indices (within bounds, [0, gridSize-1]), fetches the value from the voxel buffer.

	Note that, voxel values actually represent values at the corners not per-voxel center value in the context of Marching Cubes. In Marching Cubes, each corner actually has a SDF value. So, per voxel there are 8 values

	For example, voxels[0] is the value of voxel 0's 0'th corner's value. voxels[1] lies at one right (I store the values in z-y-x order) and so on. In other words, a voxel i's value actually means
	the values lies in its first corner. 
*/
float voxelValue(uvec3 idx)
{
	return pushConstants.voxelBuffer.voxels[idx.x + pushConstants.mcSettings.gridSize.x * (idx.y + pushConstants.mcSettings.gridSize.y * idx.z)];
}