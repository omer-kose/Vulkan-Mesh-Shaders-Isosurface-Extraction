#version 450
#extension GL_GOOGLE_include_directive : enable

#include "input_structures.h"

layout(location = 0) flat in vec2 chunkIsoValueRange;

layout(location = 0) out vec4 color;

void main()
{
    if(chunkIsoValueRange.x <= pushConstants.inputIsoValue && pushConstants.inputIsoValue <= chunkIsoValueRange.y )
    {
        color = vec4(1.0, 1.0, 0.0, 1.0);
    }
    else 
    {
        discard;
    }
}