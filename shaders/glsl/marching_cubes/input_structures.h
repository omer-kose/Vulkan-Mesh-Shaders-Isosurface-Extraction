layout(set = 0, binding = 0) uniform SceneData
{
	mat4 view;
	mat4 proj;
	mat4 viewproj;
	vec4 ambientColor;
	vec4 sunlightDirection; //w for sun power
	vec4 sunlightColor;
} sceneData;

const uint SHIFT = 6; // 128x128x128 cubes by default. (1 << 7) == 128
#define GRID_SIZE (1 << SHIFT)
#define STEP_SIZE (1.0f / float(GRID_SIZE))

// Task shader to mesh shader I/O
struct MeshletData
{
	uint MeshletID;
};

struct TaskPayload
{
	MeshletData meshlets[64];
};

// For testing purposes:
float field(vec3 pos)
{
	vec3 center = vec3(0.0, 0.0, 0.0);
	float radius = 1.0;

	// Signed distance function for a sphere
	return length(pos - center) - radius;
}