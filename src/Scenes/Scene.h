#pragma once

#include <Core/vk_types.h>

#include "camera.h"

class VulkanEngine;

/*
	Abstract base class for the scenes. Each scene must override some core functionality that is called by the engine.
*/

class Scene
{
public:
	virtual void load(VulkanEngine* engine) = 0;
	virtual void processSDLEvents(SDL_Event& e) = 0;
	virtual void handleUI() = 0; // Add UI field in ImGUI for scene parameters
	virtual void update() = 0; // called in engine update
	virtual void drawFrame(VkCommandBuffer cmd) = 0; // called by drawing logic in engine to draw the scene
	virtual GPUSceneData getSceneData(); // Returns the GPUSceneData member to update the buffer stored in the engine
	virtual ~Scene() = default;
protected:
	// Properties that exist in all the scenes
	VulkanEngine* pEngine; // Engine pointer is stored during load to be used in later when engine functionality is needed
	GPUSceneData sceneData; // All the scenes share the same scene struct properties
	Camera mainCamera;
};