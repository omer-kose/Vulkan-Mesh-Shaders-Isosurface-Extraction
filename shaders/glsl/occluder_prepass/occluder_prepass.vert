#version 450
#extension GL_GOOGLE_include_directive : enable

#include "input_structures.h"

// Hardcoded unit cube offsets from lower corner pos (12 triangles = 36 vertices). Each triplet is a triangle
const vec3 UNIT_CUBE_OFFSETS[36] = vec3[36](
    // +X face
    vec3(1.0, 0.0, 0.0),
    vec3(1.0, 1.0, 0.0),
    vec3(1.0, 1.0, 1.0),
    vec3(1.0, 0.0, 0.0),
    vec3(1.0, 1.0, 1.0),
    vec3(1.0, 0.0, 1.0),
    // -X face
    vec3(0.0, 0.0, 0.0),
    vec3(0.0, 1.0, 1.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 0.0),
    vec3(0.0, 0.0, 1.0),
    vec3(0.0, 1.0, 1.0),
    // +Y face
    vec3(0.0, 1.0, 0.0),
    vec3(1.0, 1.0, 0.0),
    vec3(1.0, 1.0, 1.0),
    vec3(0.0, 1.0, 0.0),
    vec3(1.0, 1.0, 1.0),
    vec3(0.0, 1.0, 1.0),
    // -Y face
    vec3(0.0, 0.0, 0.0),
    vec3(1.0, 0.0, 1.0),
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 0.0, 0.0),
    vec3(0.0, 0.0, 1.0),
    vec3(1.0, 0.0, 1.0),
    // +Z face
    vec3(0.0, 0.0, 1.0),
    vec3(1.0, 0.0, 1.0),
    vec3(1.0, 1.0, 1.0),
    vec3(0.0, 0.0, 1.0),
    vec3(1.0, 1.0, 1.0),
    vec3(0.0, 1.0, 1.0),
    // -Z face
    vec3(0.0, 0.0, 0.0),
    vec3(1.0, 1.0, 0.0),
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(1.0, 1.0, 0.0)
);

layout(location = 0) flat out vec3 chunkColor;

//  3 out, 1 in...
vec3 hash31(float p)
{
   vec3 p3 = fract(vec3(p) * vec3(.1031, .1030, .0973));
   p3 += dot(p3, p3.yzx+33.33);
   return fract((p3.xxy+p3.yzz)*p3.zyx); 
}

void main()
{
    // Fetch chunk data
    uint chunkID = activeChunkIndicesBuffer.activeChunkIndices[gl_InstanceIndex];
    ChunkMetadata chunkMetadata = chunkMetadataBuffer.chunkMetadata[chunkID];
    vec3 lower = chunkMetadata.lowerCornerPos;
    vec3 upper = chunkMetadata.upperCornerPos;

    vec3 scale = (upper - lower);
    vec3 worldPos = lower + UNIT_CUBE_OFFSETS[gl_VertexIndex] * scale;
    gl_Position = sceneData.viewproj * vec4(worldPos, 1.0f);
    chunkColor = hash31(float(chunkID));
}
