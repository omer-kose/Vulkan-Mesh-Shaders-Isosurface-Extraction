#include "Scene.h"

#include <Tests/SVOUnitTests.h>

/*
	A non-specific scene that can be used for any kind of unit tests
*/

class TestbedScene : public Scene
{
public:
	virtual void load(VulkanEngine* engine) override;
	virtual void processSDLEvents(SDL_Event& e) override;
	virtual void handleUI() override; // Add UI field in ImGUI for scene parameters
	virtual void update(float dt) override; // called in engine update
	virtual void drawFrame(VkCommandBuffer cmd) override; // called by drawing logic in engine to draw the scene
	virtual void performPreRenderPassOps(VkCommandBuffer cmd); // called before drawFrame to perform any operations that would like to be done after rendering 
	virtual void performPostRenderPassOps(VkCommandBuffer cmd); // called after drawFrame to perform any operations that would like to be done after rendering 
	virtual ~TestbedScene();
private:
};