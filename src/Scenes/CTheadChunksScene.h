#pragma once

#include "Scene.h"

#include <Pass/MarchingCubesPass.h>
#include <Pass/CircleGridPlanePass.h>

#include <Data/ChunkedVolumeData.h>

class CTheadChunksScene : public Scene
{
public:
	virtual void load(VulkanEngine* engine) override;
	virtual void processSDLEvents(SDL_Event& e) override;
	virtual void handleUI() override; // Add UI field in ImGUI for scene parameters
	virtual void update() override; // called in engine update
	virtual void drawFrame(VkCommandBuffer cmd) override; // called by drawing logic in engine to draw the scene
	virtual ~CTheadChunksScene();
private:
	MarchingCubesPass::MCSettings mcSettings; // Keep track of settings to be able to modify it via GUI and update once before the render
	std::unique_ptr<ChunkedVolumeData> chunkedVolumeData;
	AllocatedBuffer voxelChunksBuffer; // a pre-determined sized buffer that holds all the chunks
	VkDeviceAddress voxelChunksBufferBaseAddress;
	size_t numChunksInGpu;
};