#include "TestbedScene.h"

void TestbedScene::load(VulkanEngine* engine)
{
	SVOUnitTests svoUnitTests;
	svoUnitTests.benchmark();
}

void TestbedScene::processSDLEvents(SDL_Event& e)
{
}

void TestbedScene::handleUI()
{
}

void TestbedScene::update(float dt)
{
}

void TestbedScene::drawFrame(VkCommandBuffer cmd)
{
}

void TestbedScene::performPreRenderPassOps(VkCommandBuffer cmd)
{

}

void TestbedScene::performPostRenderPassOps(VkCommandBuffer cmd)
{

}

TestbedScene::~TestbedScene()
{
}
