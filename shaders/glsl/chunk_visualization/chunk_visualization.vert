#version 450
#extension GL_GOOGLE_include_directive : enable

#include "input_structures.h"

// Hardcoded unit cube edges (12 edges = 24 vertices)
const vec3 UNIT_CUBE_EDGES[24] = vec3[](
    // Back face (Z=0)
    vec3(0, 0, 0), vec3(1, 0, 0), // Edge 0
    vec3(1, 0, 0), vec3(1, 1, 0), // Edge 1
    vec3(1, 1, 0), vec3(0, 1, 0), // Edge 2
    vec3(0, 1, 0), vec3(0, 0, 0), // Edge 3
    // Front face (Z=1)
    vec3(0, 0, 1), vec3(1, 0, 1), // Edge 4
    vec3(1, 0, 1), vec3(1, 1, 1), // Edge 5
    vec3(1, 1, 1), vec3(0, 1, 1), // Edge 6
    vec3(0, 1, 1), vec3(0, 0, 1), // Edge 7
    // Vertical edges
    vec3(0, 0, 0), vec3(0, 0, 1), // Edge 8
    vec3(1, 0, 0), vec3(1, 0, 1), // Edge 9
    vec3(1, 1, 0), vec3(1, 1, 1), // Edge 10
    vec3(0, 1, 0), vec3(0, 1, 1)  // Edge 11
);

layout(location = 0) flat out vec2 chunkIsoValueRange;

void main()
{
    // Fetch chunk data
    ChunkVisInformation chunkInfo = pushConstants.chunkBuffer.chunks[gl_InstanceIndex];
    vec3 lower = chunkInfo.lowerCornerPos;
    vec3 upper = chunkInfo.upperCornerPos;
    chunkIsoValueRange.x = chunkInfo.minIsoValue;
    chunkIsoValueRange.y = chunkInfo.maxIsoValue;

    vec3 scale = upper - lower;
    vec3 worldPos = lower + UNIT_CUBE_EDGES[gl_VertexIndex] * scale;
    gl_Position = sceneData.viewproj * vec4(worldPos, 1.0f);
}
