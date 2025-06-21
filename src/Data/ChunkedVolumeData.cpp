#include "ChunkedVolumeData.h"

#include <Core/vk_engine.h>

ChunkedVolumeData::ChunkedVolumeData(VulkanEngine* engine, const std::vector<float>& volumeData, const glm::uvec3& gridSize_in, const glm::uvec3& chunkSize_in, const glm::vec3& gridLowerCornerPos_in, const glm::vec3& gridUpperCornerPos_in)
	:
	gridSize(gridSize_in),
	chunkSize(chunkSize_in),
	gridLowerCornerPos(gridLowerCornerPos_in),
	gridUpperCornerPos(gridUpperCornerPos_in)
{
	pEngine = engine;
	/*
		Given chunks of n voxels, there are actually n+1 points. Volume Data input consists of points, for n points there are n-1 voxels.
	*/
	numChunks = (gridSize - 1u + chunkSize - 1u) / chunkSize;
	size_t numChunksFlat = numChunks.z * numChunks.y * numChunks.x;
	size_t numPointsPerChunk = (chunkSize.z + 1) * (chunkSize.y + 1) * (chunkSize.x + 1);
	chunks.reserve(numChunksFlat);

	// Allocate the staging buffer
	size_t stagingBufferSize = numChunksFlat * numPointsPerChunk * sizeof(float);
	chunksStagingBuffer = pEngine->createBuffer(stagingBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
	pChunksStagingBuffer = (float*)pEngine->getMappedStagingBufferData(chunksStagingBuffer);
	/*
		Set all to some constant. This guarantees that, the chunks that is partly outside of the grid (if the size along some axis is not multiple of chunk size) get a default value
	*/
	for(size_t i = 0; i < numChunksFlat * numPointsPerChunk; ++i)
	{
		pChunksStagingBuffer[i] = -FLT_MAX;
	}

	// Divide volume into chunks
	for(size_t z = 0; z < numChunks.z; ++z)
	{
		for(size_t y = 0; y < numChunks.y; ++y)
		{
			for(size_t x = 0; x < numChunks.x; ++x)
			{
				VolumeChunk chunk;
				chunk.chunkIndex = glm::uvec3(x, y, z);
				extractChunkData(volumeData, x + numChunks.x * (y + numChunks.y * z), chunk);
				chunks.push_back(chunk);
			}
		}
	}

	// Construct the interval tree
	std::vector<VolumeChunk*> chunkAddresses(numChunksFlat);
	for(int i = 0; i < numChunksFlat; ++i)
	{
		chunkAddresses[i] = &chunks[i];
	}

	intervalTree.build(chunkAddresses);
}

std::vector<VolumeChunk*> ChunkedVolumeData::query(float isoValue) const
{
	return intervalTree.query(isoValue);
}

glm::uvec3 ChunkedVolumeData::getNumChunks() const
{
	return numChunks;
}

glm::uvec3 ChunkedVolumeData::getChunkSize() const
{
	return chunkSize;
}

VkBuffer ChunkedVolumeData::getStagingBuffer() const
{
	return chunksStagingBuffer.buffer;
}

ChunkedVolumeData::~ChunkedVolumeData()
{
	pEngine->destroyBuffer(chunksStagingBuffer);
}

/*
	Extracts and writes the data of the chunk into the staging buffer
*/
void ChunkedVolumeData::extractChunkData(const std::vector<float>& volumeData, size_t flatChunkIndex, VolumeChunk& chunk)
{
	// Compute the bound indices of the chunk for fetching the data from volume data. (Mapping from chunk index to actual grid index)
	glm::uvec3 startIndex = chunkSize * chunk.chunkIndex;
	glm::uvec3 endIndex = glm::min(startIndex + chunkSize + 1u, gridSize); // exclusive end, +1 as there are n+1 points in a chunk of n voxels
	
	// TODO: If chunk size is 32 for example, there are actually 33 points in any axis. Therefore, both allocation and processing limits are wrong. Fix that. It should be quite simple just alloc chunkSize + 1 and process accordingly. On GPU side as well

	// TODO: Check this after fixing the thing above
	// Compute the lower and upper corner positions of the chunk from the grid corners.
	glm::vec3 stepSize = (gridUpperCornerPos - gridLowerCornerPos) / glm::vec3(gridSize - 1u);
	chunk.lowerCornerPos = gridLowerCornerPos + glm::vec3(startIndex) * stepSize;
	chunk.upperCornerPos = chunk.lowerCornerPos + glm::vec3(chunkSize) * stepSize;

	chunk.minIsoValue = FLT_MAX;
	chunk.maxIsoValue = -FLT_MAX;

	// Compute and store the offset of the given chunk in the staging buffer
	size_t chunkVoxelsFlatIndex = flatChunkIndex * ((chunkSize.x + 1) * (chunkSize.y + 1) * (chunkSize.z + 1)); // the actual starting index of the voxels of the chunk in the staging buffer
	chunk.stagingBufferOffset = chunkVoxelsFlatIndex * sizeof(float);
	
	float* pChunkVoxels = pChunksStagingBuffer + chunkVoxelsFlatIndex;
	
	glm::uvec3 pointsPerAxis = chunkSize + 1u;

	for(size_t z = startIndex.z; z < endIndex.z; ++z)
	{
		for(size_t y = startIndex.y; y < endIndex.y; ++y)
		{
			for(size_t x = startIndex.x; x < endIndex.x; ++x)
			{
				float val = volumeData[x + gridSize.x * (y + gridSize.y * z)];
				chunk.minIsoValue = std::min(chunk.minIsoValue, val);
				chunk.maxIsoValue = std::max(chunk.maxIsoValue, val);
				// Write the value into the correct position in the staging buffer
				glm::uvec3 localIdx = glm::uvec3(x, y, z) - startIndex;
				pChunkVoxels[localIdx.x + pointsPerAxis.x * (localIdx.y + pointsPerAxis.y * localIdx.z)] = val;
			}
		}
	}
}
