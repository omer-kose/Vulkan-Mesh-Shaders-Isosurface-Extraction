# Isosurface Extraction with Vulkan Mesh Shaders

This is the implementation of my Master's Thesis "Isosurface Extraction with Vulkan Mesh Shaders" at Technical University of Munich. 

## Development Environment
- C++
- Vulkan 1.3 (with [VK Bootstrap](https://github.com/charles-lunarg/vk-bootstrap), [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) and [volk](https://github.com/gnuradio/volk))
- GLSL
- [GLM](https://github.com/g-truc/glm)
- [SDL](https://github.com/libsdl-org/SDL)
- [ImGui](https://github.com/ocornut/imgui)
- [fmt](https://github.com/fmtlib/fmt)
- [Ogt Vox](https://github.com/jpaver/opengametools/blob/master/src/ogt_vox.h) for loading .vox files

## Implementation Details  

- **GPU-Driven Mesh Shader Pipeline**  
  A fully GPU-driven indirect mesh shader pipeline is implemented, supporting both **frustum culling** and **Hierarchical Z-Buffer occlusion culling** entirely in compute shaders.  

- **Volume-Based Input (No Pre-Extracted Meshes)**  
  Unlike traditional pipelines that pre-extract and store triangle meshes, this implementation takes a **3D volume** as input—either an **isosurface SDF field** or an **occupancy field**. No explicit mesh storage is required.  

- **On-the-Fly Isosurface Extraction with Marching Cubes**  
  For SDF inputs, a **task and mesh shader optimized Marching Cubes algorithm** is used. Triangles are extracted directly on the GPU and rendered in a **single pass per frame**. The design is scalable to any 3D input size that fits in GPU memory. (Inspired by [Emil Persson’s work](https://www.humus.name/index.php?page=3D)).  

- **Voxel Rendering Without Triangle Extraction**  
  For occupancy fields (voxel data), a dedicated **voxel renderer** is implemented using task and mesh shaders. Hidden surfaces are culled, and visible voxel faces are directly rasterized in a **single pass**—again, without storing any explicit triangles.  

- **Chunk-Based Meshlets**  
  The 3D input data is divided into **chunks**, which serve as meshlets for rendering. Chunks provide multiple benefits:  
  - **Early elimination** of entire chunks for SDFs when the isovalue range of the chunk is outside the input isovalue.  
  - **Bounding volumes** used for efficient frustum and occlusion culling.  
  - **Cache-friendly layout**, as chunks are stored contiguously in GPU memory, enabling reordering and improved memory access patterns.
  - **Finer-Block based processing** can be done thanks to linearity of the chunks. Without any overhead, chunks are actually processed in finer 4x4x4 blocks allowing more precise culling and efficient utilization of the shader pipeline. 

- **Sparse Voxel Octree with Bricks (Voxel Rendering)**  
  For voxel rendering, a **brick-based sparse voxel octree** is implemented:  
  - Bricks are **4×4×4** in size, providing fine-grained elimination of empty space.  
  - Greatly improves **memory efficiency** and enables **level-of-detail (LOD)** rendering.  
  - Prevents processing of empty air regions, further accelerating the pipeline.  

## Results


### Marching Cubes Isosurface Extraction

| ![Head](https://github.com/user-attachments/assets/7bb0c2e1-d95e-4f53-83b7-72e623460854) | ![Kidney](https://github.com/user-attachments/assets/4ec444e3-134d-4c2b-932f-365c0cc72419) |
|:--:|:--:|
| *CT Head Visualization (256×256×113)* | *Kidney Visualization (629×515×955)* |

The kidney dataset highlights a key challenge of **Marching Cubes**: as input resolution increases, the algorithm produces a large number of tiny triangles, creating a bottleneck since millions of small primitives must be rendered.  

To address this, the pipeline leverages **GPU-driven frustum and occlusion culling** on finer 4x4x4 blocks of the chunks. Only visible blocks are dispatched indirectly through the mesh shader pipeline, significantly reducing the rendering workload and improving performance.


