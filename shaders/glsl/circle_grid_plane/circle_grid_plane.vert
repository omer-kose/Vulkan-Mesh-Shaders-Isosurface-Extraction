#version 450
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : require
#include "input_structures.h"

layout(location = 0) out vec3 fragWorldPos;

const vec3 vertices[6] = vec3[6](
    vec3(-1, 0, -1),  // Triangle 1
    vec3( 1, 0, -1),
    vec3(-1, 0,  1),
    vec3( 1, 0, -1),  // Triangle 2
    vec3( 1, 0,  1),
    vec3(-1, 0,  1)
);

void main()
{
    vec3 localPos = vertices[gl_VertexIndex];
    localPos.y += pushConstants.planeHeight;

    float planeScale = 10.0;
    localPos *= planeScale;

    // Center the plane around origin
    vec3 worldPos = localPos;

    // Output
    gl_Position = sceneData.viewproj * vec4(worldPos, 1.0);
    fragWorldPos = worldPos;
}