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
	void loadData(uint32_t organID);
	std::pair<AllocatedBuffer, glm::uvec3> loadCTheadData() const;
	std::pair<AllocatedBuffer, glm::uvec3> loadOrganAtlasData(const char* organPathBase);
private:
	// Data Loading Params
	std::vector<std::string> organNames; // This is for selecting the organ data from UI. The names are hardcoded. 
	uint32_t selectedOrganID; // Keep track of the current data ID to see if the data is changed.

	MarchingCubesPass::MCSettings mcSettings; // Keep track of settings to be able to modify it via GUI and update once before the render
	AllocatedBuffer voxelBuffer;
};