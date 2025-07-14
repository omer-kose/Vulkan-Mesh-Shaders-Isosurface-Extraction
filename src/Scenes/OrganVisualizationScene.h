#pragma once

#include "Scene.h"

#include <Pass/MarchingCubesPass.h>
#include <Pass/CircleGridPlanePass.h>

class OrganVisualizationScene : public Scene
{
public:
	virtual void load(VulkanEngine* engine) override;
	virtual void processSDLEvents(SDL_Event& e) override;
	virtual void handleUI() override; // Add UI field in ImGUI for scene parameters
	virtual void update() override; // called in engine update
	virtual void drawFrame(VkCommandBuffer cmd) override; // called by drawing logic in engine to draw the scene
	virtual ~OrganVisualizationScene();
private:
	std::pair<AllocatedBuffer, glm::uvec3> loadCTheadData() const;
private:
	MarchingCubesPass::MCSettings mcSettings; // Keep track of settings to be able to modify it via GUI and update once before the render
	AllocatedBuffer voxelBuffer;
};