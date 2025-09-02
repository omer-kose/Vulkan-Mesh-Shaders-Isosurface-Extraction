#extension GL_GOOGLE_include_directive : enable

#include "../global_input.h"

layout(push_constant, scalar) uniform PushConstants
{
	float planeHeight;
} pushConstants;