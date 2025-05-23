#version 450

layout (location = 0) out vec4 color;

layout(location = 0) in MeshIn
{
	vec3 normal;
} meshIn;

void main()
{
  color = vec4(meshIn.normal,1.0);
}