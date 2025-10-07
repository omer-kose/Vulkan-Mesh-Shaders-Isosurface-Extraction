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
  For SDF inputs, a **task and mesh shader optimized Marching Cubes algorithm** is used. Triangles are extracted directly on the GPU and rendered in a **single pass per frame**. The design is scalable to any 3D input size that fits in GPU memory. (Inspired by [Emil Persson’s work](https://www.humus.name/index.php?page=3D&ID=93)).  

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
  - LOD selection is done in an asynchronous worker thread using double buffering. Each time main render thread requests, a result is ready which prevents CPU spikes when the camera moves.

## Results and Analysis

### Marching Cubes Isosurface Extraction

<table>
  <tr>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/head-video-id.mp4](https://github.com/user-attachments/assets/54ff955c-f53a-450c-b757-b121e1bf26d4" width="320" controls></video>
      <br>
      <sub>CT Head Visualization (256×256×113)</sub>
    </td>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/e5e343df-bea6-4472-8e2f-588dae2b0a86" width="320" controls></video>
      <br>
      <sub>Kidney Visualization (629×515×955)</sub>
    </td>
  </tr>
</table>

The kidney data highlights a key challenge of **Marching Cubes**: as input resolution increases, the algorithm produces a large number of tiny triangles, creating a bottleneck since millions of small primitives must be rendered.  

To address this, the pipeline leverages **GPU-driven frustum and occlusion culling** on finer 4x4x4 blocks of the chunks. Only visible blocks are dispatched indirectly through the mesh shader pipeline, significantly reducing the rendering workload and improving performance. Task shaders processes those 4x4x4 blocks efficiently and dispatches mesh shader groups per voxels. Each mesh shader group, processes a single voxel and extracts required information, including normals for free by using subgroup operations, from the Marching Cubes table. 

### Voxel Rendering

<table>
  <tr>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/23e3c257-bd9a-4183-911c-51d8b4c4894a" width="320" controls></video>
      <br>
      <sub>Generated Voxel Field (1024×1024×1024)</sub>
    </td>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/e9d03e92-67d4-44f2-9241-6722d38475e8" width="320" controls></video>
      <br>
      <sub>Generated Voxel Field (2048×1024×2048)</sub>
    </td>
  </tr>
</table>

This demonstrates the performance of the pipeline on very dense voxel fields, using **GPU-based culling and LOD**, on a compact range to make the entire field visible for limit-testing.  

Voxel pipeline workflow is as follows:

1. **Brick Based Sparse Voxel Octree Extraction:**  
   A brick-based sparse voxel octree is extracted from the dense field, and its GPU nodes are uploaded once.  

2. **LOD Selection:**  
   A worker thread computes LOD selection in parallel to render thread, producing a flat list of node indices for the compute shader. In the compute shader, frustum and occlusion culling is applied, so only visible nodes are dispatched indirectly.

3. **Task Shader Processing:**  
   - **LOD0:** Bricks are processed at the voxel level for high-detail rendering.  
   - **LOD1:** Immediate parent bricks (coarse **2×2×2** bricks created by downsampling) are extracted and rendered by the task shader.  
   - **LOD2+ :** Higher-level bricks are collapsed into single nodes, rendered as individual cubes.  

Despite the extreme density (up to **4 billion voxels** in a compact area), the pipeline achieves **≥60 FPS** on average without any explicit meshing and purely based on voxel data. Normally, fields of this density would span kilometers in a real-world scenario, but testing them in a compact space stresses the rendering system and demonstrates the efficiency of the GPU-driven Voxel pipeline.

## References
- The engine underwent lots of changes but the baseline is built following: [vkguide](https://vkguide.dev/)
- [Emil Persson's Efficient Implementation of Marching Cubes in Task and Mesh Shaders](https://www.humus.name/index.php?page=3D)
- [Niagara's GPU based Frustum and Occlusion Pipeline](https://github.com/zeux/niagara)
- [Projecting Bounding Box to NDC for Hierarchical Z-Buffer Occlusion Culling](https://zeux.io/2023/01/12/approximate-projected-bounds/)
- [Converting the CTHead data](https://github.com/keijiro/ComputeMarchingCubes/blob/main/Assets/VolumeData/VolumeDataConverter.compute)
- [Organ Atlas](https://human-organ-atlas.esrf.fr/)



