#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <span>
#include <array>
#include <functional>
#include <deque>

#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vk_mem_alloc.h>

#include <fmt/core.h>

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

#define VK_CHECK(x)                                                     \
    do {                                                                \
        VkResult err = x;                                               \
        if (err) {                                                      \
            fmt::println("Detected Vulkan error: {}", string_VkResult(err)); \
            abort();                                                    \
        }                                                               \
    } while (0)


struct AllocatedImage
{
    VkImage image;
    VkImageView imageView;
    VmaAllocation allocation;
    VkExtent3D imageExtent;
    VkFormat imageFormat;
};

struct AllocatedBuffer
{
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo allocInfo;
};

// Layout of vertex (storage) buffer
struct Vertex
{
    glm::vec3 position;
    float uv_x;
    glm::vec3 normal;
    float uv_y;
    glm::vec4 color;
};

// Holds the resources needed for a mesh
struct GPUMeshBuffers
{
    AllocatedBuffer indexBuffer;
    AllocatedBuffer vertexBuffer;
    VkDeviceAddress vertexBufferAddress;
};

// Push constants for mesh object draws
struct GPUDrawPushConstants
{
    glm::mat4 worldMatrix;
    VkDeviceAddress vertexBufferAddress;
};

// All the scenes need the same common information on descriptor set 0 and this struct will only extend if all the scenes need that
struct GPUSceneData
{
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewproj;
    glm::mat4 prevViewProj; // Previous frame's view projection. Might be useful when temporal reprojection is needed
    glm::vec4 ambientColor;
    glm::vec4 sunlightDirection; // w for sun power
    glm::vec4 sunlightColor;
    glm::vec3 cameraPos;
};

struct DrawContext;

// Base class for renderable dynamic object
class IRenderable
{
    virtual void registerDraw(const glm::mat4& topMatrix, DrawContext& ctx) = 0;
};

/*
    GLTF Scene Node
    The scene node can hold children and will also keep a transform to propagate to them
*/
struct GLTFSceneNode : public IRenderable
{
    // parent pointer must be a weak ptr to avoid the circular dependency problem
    std::weak_ptr<GLTFSceneNode> parent;
    std::vector<std::shared_ptr<GLTFSceneNode>> children;

    glm::mat4 localTransform;
    glm::mat4 worldTransform; // actual worldMatrix (or model matrix)
    
    // Must be called recursively whenever the localTransform is changed
    void refreshTransform(const glm::mat4& parentMatrix)
    {
        worldTransform = parentMatrix * localTransform;
        for(auto c : children)
        {
            c->refreshTransform(worldTransform);
        }
    }

    virtual void registerDraw(const glm::mat4& topMatrix, DrawContext& ctx)
    {
        // draw children 
        for(auto c : children)
        {
            c->registerDraw(topMatrix, ctx);
        }
    }
};