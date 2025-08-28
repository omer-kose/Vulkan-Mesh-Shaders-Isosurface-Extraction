#pragma once

#include <Core/vk_types.h>
#include <Core/vk_descriptors.h>
#include <Core/vk_loader.h>

#include <Pass/CircleGridPlanePass.h>
#include <Pass/ChunkVisualizationPass.h>
#include <Pass/MarchingCubesPassSDF.h>
#include <Pass/MarchingCubesPass.h>
#include <Pass/HZBDownSamplePass.h>
#include <Pass/MarchingCubesIndirectPass.h>
#include <Pass/VoxelRenderingIndirectPass.h>
#include <Pass/OccluderPrePass.h>

#include <Scenes/OrganVisualizationScene.h>
#include <Scenes/VoxelRenderingScene.h>

struct DeletionQueue
{
	void pushFunction(std::function<void()>&& function)
	{
		deletors.push_back(function);
	}

	void flush()
	{
		// Reverse iterate the deletion queue to execute all the deletion functions
		for(auto it = deletors.rbegin(); it != deletors.rend(); ++it)
		{
			(*it)();
		}

		deletors.clear();
	}

	std::deque<std::function<void()>> deletors;

};

struct FrameData
{
	VkCommandPool commandPool;
	VkCommandBuffer mainCommandBuffer;

	VkSemaphore swapchainSemaphore, renderSemaphore;
	VkFence renderFence;
	
	DescriptorAllocatorGrowable frameDescriptorAllocator;

	// Per-Frame Resource Deletion Queue
	DeletionQueue deletionQueue;
};

struct EngineStats
{
	float frameTime;
	int triangleCount;
	int drawCallCount;
	float sceneUpdateTime;
	float geometryDrawRecordTime;
};

constexpr unsigned int FRAME_OVERLAP = 2;

/*
	Represents the geometry (and a possible material instance) of an object to be drawn in that frame. Created and destroyed per-frame. 

	It can represent geometry from all kinds of formats.
*/
struct RenderObject
{
	uint32_t indexCount;
	uint32_t firstIndex;
	VkBuffer indexBuffer;

	MaterialInstance* materialInstance; // a non-owning pointer

	Bounds bounds;

	glm::mat4 transform;
	VkDeviceAddress vertexBufferAddress;
};

/*
	Holds a flat list objects to be drawn that frame. The list is filled and reset every frame.

	For the time being, meshes coming from different formats are held in different lists so that the related passes can only fetch the required meshes and work with them.
*/
struct DrawContext
{
	std::vector<RenderObject> opaqueGLTFSurfaces;
	std::vector<RenderObject> transparentGLTFSurfaces;
};

class VulkanEngine
{
public:
	static VulkanEngine& Get();

	// initializes everything in the engine
	void init();

	// shuts down the engine
	void cleanup();

	// draw functionality
	void draw(); // core draw loop
	void drawMain(VkCommandBuffer cmd); // function to simplify the main draw function. It handles some transitions, attachments and calls to actualy drawing functionality below
	void drawGeometry(VkCommandBuffer cmd);
	void drawImgui(VkCommandBuffer cmd, VkImageView targetImageView); 

	// Updates
	void updateScene();

	//run main loop
	void run();

	void immediateSubmit(std::function<void(VkCommandBuffer cmd)>&& function);

	// Engine utilities (TODO: For the time being most of the stuff are directly open to outside but I will be slowly hiding them)
	AllocatedBuffer createBuffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
	// Allocates a buffer on local device memory and uploads the given data using a staging buffer
	AllocatedBuffer createAndUploadGPUBuffer(size_t allocSize, VkBufferUsageFlags usage, const void* data, size_t srcOffset = 0, size_t dstOffset = 0);
	// Allocates a buffer on local device memory and uploads an already existing staging buffer on the CPU
	AllocatedBuffer uploadStagingBuffer(VkBuffer stagingBuffer, size_t allocSize, VkBufferUsageFlags usage, size_t srcOffset = 0, size_t dstOffset = 0);
	AllocatedBuffer downloadGPUBuffer(VkBuffer gpuBuffer, size_t allocSize, size_t srcOffset = 0, size_t dstOffset=0);
	VkDeviceAddress getBufferDeviceAddress(VkBuffer buffer);
	void* getMappedStagingBufferData(const AllocatedBuffer& buffer);
	void destroyBuffer(const AllocatedBuffer& buffer);

	AllocatedImage createImage(VkExtent3D imageExtent, VkFormat format, VkImageUsageFlags usage, bool mipMapped = false);
	AllocatedImage createImage(void* data, VkExtent3D imageExtent, VkFormat format, VkImageUsageFlags usage, bool mipMapped = false);
	VkImageView createImageView(VkImage image, VkFormat format, uint32_t mipLevel, uint32_t levelCount);
	VkSampler createImageSampler(VkFilter filter, VkSamplerMipmapMode mipmapMode, VkSamplerAddressMode addressMode, VkSamplerReductionModeEXT reductionMode);
	void destroyImage(const AllocatedImage& img);

	GPUMeshBuffers uploadMesh(std::span<Vertex> vertices, std::span<uint32_t> indices);

	void updateSceneBuffer();
	VkDescriptorSetLayout getSceneDescriptorLayout() const;
	VkDescriptorSet getSceneBufferDescriptorSet() const;

	void setViewport(VkCommandBuffer cmd);
	void setScissor(VkCommandBuffer cmd);

	const DrawContext* getDrawContext() const;

	VkExtent2D getWindowExtent() const;
	void setColorAttachmentClearColor(const VkClearValue& clearValue);

public:
	struct SDL_Window* window{ nullptr };

	bool isInitialized{ false };
	uint32_t frameNumber{0};
	bool freezeRendering{ false };
	bool resizeRequested{ false };
	float renderScale{ 1.0f };
	VkExtent2D windowExtent{ 1920 , 1080 };
	// Vulkan Context
	VkInstance instance; // Vulkan library handle
	VkDebugUtilsMessengerEXT debugMessenger; // Vulkan debug output handle
	VkPhysicalDevice chosenGPU; // GPU chosen as the default device
	VkDevice device; // vulkan logical device for commands.
	VkSurfaceKHR surface; // Vulkan window surface
	
	// Swapchain 
	VkSwapchainKHR swapchain;
	VkFormat swapchainImageFormat;
	std::vector<VkImage> swapchainImages;
	std::vector<VkImageView> swapchainImageViews;
	VkExtent2D swapchainExtent;

	// Queues
	VkQueue graphicsQueue;
	uint32_t graphicsQueueFamily;

	// Allocator
	VmaAllocator vmaAllocator;

	// Frame Data and Queues
	FrameData frames[FRAME_OVERLAP];
	FrameData& getCurrentFrame() { return frames[frameNumber % FRAME_OVERLAP]; }

	// Global Resource Deletion Queue
	DeletionQueue mainDeletionQueue;

	// Engine stats
	EngineStats stats;

	// Immeadiate submit structures
	VkFence immeadiateFence;
	VkCommandPool immediateCommandPool;
	VkCommandBuffer immediateCommandBuffer;

	// Main color attachment clear value 
	VkClearValue colorAttachmentClearValue = { 0.0f, 0.0f, 0.0f, 1.0f };

	// Draw Image
	AllocatedImage drawImage;
	AllocatedImage depthImage;
	VkExtent2D drawExtent;

	// Global Descriptors
	DescriptorAllocatorGrowable globalDescriptorAllocator;
	// Main draw image descriptor used as the primary render target
	VkDescriptorSetLayout drawImageDescriptorSetLayout;
	VkDescriptorSet drawImageDescriptorSet;
	// Descriptor layout for single texture display
	VkDescriptorSetLayout displayTextureDescriptorSetLayout;

	// Per-frame Global Scene (uniform) Buffer and the descriptor set (Shared by the whole engine which uses scene data so it is persistent per-frame no need to reallocate) 
	// All the scenes have the same common structure for the buffer so resource is stored in the engine not in the active scene
	AllocatedBuffer gpuSceneDataBuffer[FRAME_OVERLAP];
	VkDescriptorSet sceneDescriptorSet[FRAME_OVERLAP];
	VkDescriptorSetLayout sceneDescriptorLayout;

	// Default textures
	AllocatedImage whiteImage;
	AllocatedImage blackImage;
	AllocatedImage greyImage;
	AllocatedImage errorCheckerboardImage;
	
	// Default samplers
	VkSampler defaultSamplerLinear;
	VkSampler defaultSamplerNearest;	

	// Default materials
	MaterialInstance defaultMaterialInstance;

	// Main Draw Context
	DrawContext mainDrawContext;

	// Scene
	std::unique_ptr<Scene> activeScene;
	std::vector<std::string> sceneNames; // This is for selecting the scenes from UI. The names are hardcoded. 
	uint32_t selectedSceneID; // Keep track of the current scene name to see if the scene is changed.

private:
	// Vulkan Context
	void m_initVulkan();
	void m_initSwapchain();
	void m_initCommands();
	void m_initSyncStructures();
	// Swapchain
	void m_createSwapchain(uint32_t width, uint32_t height);
	void m_destroySwapchain();
	void m_resizeSwapchain();
	// Descriptors
	void m_initDescriptors();

	// Passes
	void m_initPasses();
	void m_clearPassResources();

	// Material Layouts
	void m_initMaterialLayouts(); 
	void m_clearMaterialLayouts();

	// ImGui
	void m_initImgui();

	// Default Engine Data
	void m_initDefaultData();

	// Init Scene Buffer
	void m_initGlobalSceneBuffer();

	// Scene
	void m_initSceneInformation();
	void loadScene(uint32_t sceneID);
};
