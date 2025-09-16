#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_GOOGLE_include_directive : enable

#include "../global_input.h"

struct VoxelColor
{
	uint8_t color[4]; // also compatible with vox files
};

layout(set = 1, binding = 0, scalar) uniform Palette
{
	VoxelColor palette[256];
};

#define BRICK_SIZE 8  // brick size of the leaf 

// Derived Constants
#define BRICK_PLUS_1 (BRICK_SIZE + 1)  // N+1 samples needed for N cubes
#define BRICK_VOLUME (BRICK_SIZE * BRICK_SIZE * BRICK_SIZE)

/*
	GPU version of SVO nodes. For GPU-side, each node is nothing but a voxel with varying information required. Those voxels are not uniform and can be from different levels of the hierarchy
*/
struct SVONodeGPU
{
	vec3 lowerCorner;
	vec3 upperCorner;
	uint8_t   colorIndex;
	uint8_t   level;      // 0 = finest voxels (we use bricks at leafLevel)
	uint32_t  brickIndex; // UINT32_MAX => no brick present (mono-color leaf or internal)
};

// simple fixed-size brick type (stores BRICK_SIZE^3 bytes)
struct Brick
{
	uint8_t voxels[BRICK_VOLUME];
};

/*
	Necessary information for the task and mesh shaders to be able to fetch required chunk data for dispatch. This will be filled by the compute shader for each task shader dispatch.

	Each task shader invocation has a unique NodeDrawData entry
*/
struct NodeDrawData
{
	uint nodeID; // ID of the chunk in the ChunkMetadata array
};

layout(buffer_reference, scalar) readonly buffer SVONodeGPUBuffer
{
	SVONodeGPU nodes[]; // Contains all the GPU nodes of the SVO tree
};

layout(buffer_reference, scalar) readonly buffer BrickBuffer
{
	Brick bricks[]; // Contains color data. If 0 that means Voxel is not solid (not occupied).
};


layout(buffer_reference, scalar) buffer NodeDrawDataBuffer
{
	ChunkDrawData nodeDrawData[];
};

layout(buffer_reference, scalar) readonly buffer ActiveNodeIndicesBuffer
{
	uint indices[];
};

/*
	Final number of nodes that are guaranteed to be processed by task shaders.

	2 dummy values for y and z dispatches which will be always zero. Putting them there to match with the VkDrawMeshTasksIndirectCommandEXT struct as this will be also used as the indirect buffer
*/
layout(buffer_reference, scalar) buffer DrawNodeCountBuffer
{
	uint drawNodeCount;
	uint _dummy1;
	uint _dummy2;
};

layout(push_constant, scalar) uniform PushConstants
{
	uint numActiveNodes; // Number of nodes to process this frame. Not the total number of nodes. Practically active size of the nodeIndices bufer
	uint leafLevel; // TODO: Try making this uint8_t. Should work I believe
	float zNear;
	uint depthPyramidWidth;
	uint depthPyramidHeight;
	SVONodeGPUBuffer svoNodeGPUBuffer;
	BrickBuffer brickBuffer;
	NodeDrawDataBuffer nodeDrawDataBuffer;
	DrawNodeCountBuffer drawNodeCountBuffer;
	ActiveNodeIndicesBuffer activeNodeIndicesBuffer;
};

// Task shader to mesh shader. Represents a single voxel meshlet in the brick.
struct MeshletData
{
	uint meshletID;
	uint colorIndex; // TODO: This could be a 1 byte uint but I couldn't get it to working
};

struct TaskPayload
{
	// These two are only filled if task shader processes a leaf node
	MeshletData meshlets[BRICK_VOLUME]; 
	uint nodeColorIndex; // If non-leaf node has a single representative color. Only filled if task shader is processing a non-leaf node
	uint nodeID;
};

/*
	Fetches the voxel value that stores color value per voxel from the brick that leaf node represents.
*/
uint voxelValue(uint nodeID, uvec3 idx)
{
	uint brickIdx = svoNodeGPUBuffer.nodes[nodeID].brickIndex;
	return uint(brickBuffer.bricks[brickIdx].voxels[idx.x + shellSize.x * (idx.y + shellSize.y * idx.z)]);
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


