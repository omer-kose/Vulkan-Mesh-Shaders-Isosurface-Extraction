#extension GL_EXT_scalar_block_layout : require
#extension GL_GOOGLE_include_directive : enable

#include "../global_input.h"

#define BLOCK_SIZE 4  // block size that each group processes (e.g., 4x4x4)

// Derived Constants
#define BLOCK_PLUS_1 (BLOCK_SIZE + 1)  // N+1 samples needed for N cubes
#define BLOCK_VOLUME (BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE)

layout(set = 1, binding = 1, scalar) uniform MCSettings
{
	uvec3 gridSize;
} mcSettings;

// Task shader to mesh shader I/O
struct MeshletData
{
	uint meshletID;
};

struct TaskPayload
{
	MeshletData meshlets[64];
};

// For testing purposes:
float field(vec3 pos)
{
	// Unit sphere centered in [0,1] grid
	vec3 center = vec3(0.5);
	float radius = 0.25;
	return length(pos - center) - radius;
}