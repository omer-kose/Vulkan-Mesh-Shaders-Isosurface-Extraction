#version 450
#extension GL_GOOGLE_include_directive : enable

#include "input_structures.h"

layout(location = 0) out vec4 color;

layout(location = 0) flat in vec3 chunkColor;

void main()
{
    // Only depth write is important so color here is not important
    color = vec4(chunkColor, 1.0);
}