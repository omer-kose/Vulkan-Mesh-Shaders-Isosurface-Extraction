#extension GL_EXT_scalar_block_layout : require

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