#pragma once 

#include <vulkan/vulkan.h>

namespace vkutil 
{
	void transitionImage(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout);

	void copyImageToImage(VkCommandBuffer cmd, VkImage source, VkImage destination, VkExtent2D srcSize, VkExtent2D destSize);

	void generateMipmaps(VkCommandBuffer cmd, VkImage image, VkExtent2D imageSize);
};