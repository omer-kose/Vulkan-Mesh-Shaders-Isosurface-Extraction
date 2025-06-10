#version 450
#extension GL_GOOGLE_include_directive : require
#include "input_structures.h"

layout(location = 0) in vec3 fragWorldPos;
layout(location = 0) out vec4 outFragColor;

void main() 
{
    // Grid parameters
    float gridSize = 0.7;
    float lineWidth = 0.005;
    
    // Calculate grid lines
    vec2 coord = fragWorldPos.xz / gridSize;
    vec2 grid = abs(fract(coord - 0.5) - 0.5);
    float line = min(grid.x, grid.y);
    float fValue = fwidth(line);
    float isLine = smoothstep(lineWidth - fValue, lineWidth + fValue, line);
    
    // Colors
    vec3 lineColor = vec3(0.0); // Black lines
    vec3 gridColor = vec3(0.75); // Silver background
    
    // Circular fade parameters
    float visibleRadius = 5.0; // Radius of full visibility
    float fadeRadius = 5.3;    // Radius where fade completes
    
    // Calculate distance from camera
    float dist = length(fragWorldPos);
    
    // Circular fade calculation (1.0 at center, 0.0 beyond fadeRadius)
    float fade = 1.0 - smoothstep(visibleRadius, fadeRadius, dist);
    
    // Apply circular mask
    vec3 finalColor = mix(lineColor, gridColor, isLine) * fade;
    
    // Final output with alpha
    outFragColor = vec4(finalColor, fade);
}