# Isosurface Extraction with Vulkan Mesh Shaders

This is the implementation of my Master's Thesis "Isosurface Extraction with Vulkan Mesh Shaders" at Technical University of Munich. 

## Development Environment
- C++
- Vulkan 1.3 (with [VK Bootstrap](https://github.com/charles-lunarg/vk-bootstrap), [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) and [volk](https://github.com/gnuradio/volk))
- GLSL
- [GLM](https://github.com/g-truc/glm)
- [SDL](https://github.com/libsdl-org/SDL)
- [ImGui](https://github.com/ocornut/imgui)
- [https://github.com/fmtlib/fmt](https://github.com/fmtlib/fmt)
- [Ogt Vox](https://github.com/jpaver/opengametools/blob/master/src/ogt_vox.h) for loading .vox files

## Implementation Details
- No triangle meshes are pre-extracted and stored explicitly. The input is a 3D volume, either an isosurface SDF field or an occupancy field.
- The triangles are extracted using Task and Mesh shaders from the input isosurface efficiently. 
