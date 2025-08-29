#include "ChunkedVolumeData.h"

#include <Core/vk_engine.h>

#include <Pass/MarchingCubesLookup.h>

#include <algorithm>
#include <execution>

ChunkedVolumeData::ChunkedVolumeData(VulkanEngine* engine, const std::vector<uint8_t>& volumeData, const glm::uvec3& gridSize_in, const glm::uvec3& chunkSize_in, const glm::vec3& gridLowerCornerPos_in, const glm::vec3& gridUpperCornerPos_in)
	:
	gridSize(gridSize_in),
	chunkSize(chunkSize_in),
	gridLowerCornerPos(gridLowerCornerPos_in),
	gridUpperCornerPos(gridUpperCornerPos_in)
{
	pEngine = engine;
	/*
		Chunk size determines how many points on each axis of a chunk. Each point corresponds to the top-left-back point of the voxel. So, chunk size of n means, n top-left-back points and thus n voxels.
		However, for the bottom-right-front boundary, right neighbouring value is needed to be able reconstruct triangles in that voxel so that value is also read, thus +1.
		Moreover, to be able to compute normals consistently (as I am using forward differences for normals), the right neighbouring value of that right neighbour is needed, thus another +1.
		So, each chunk actualy contains a +2 shell on their bottom-right-front boundaries for correct reconstruction. 

		Note: the last +1 for the normals could be alleviated by computing the normals using backward differences on the boundaries. For consistency, I am having +2.
	*/
	numChunks = (gridSize + chunkSize - 1u) / chunkSize;
	size_t numChunksFlat = numChunks.z * numChunks.y * numChunks.x;
	size_t numPointsPerChunk = (chunkSize.z + 2) * (chunkSize.y + 2) * (chunkSize.x + 2);

	// Allocate the staging buffer
	size_t stagingBufferSize = numChunksFlat * numPointsPerChunk * sizeof(uint8_t);
	chunksStagingBuffer = pEngine->createBuffer(stagingBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
	pChunksStagingBuffer = (uint8_t*)pEngine->getMappedStagingBufferData(chunksStagingBuffer);
	std::memset(pChunksStagingBuffer, 0, stagingBufferSize);

	chunks.resize(numChunksFlat);
	std::for_each(std::execution::par, chunks.begin(), chunks.end(), [&](VolumeChunk& chunk) {
			// Compute flat index from pointer difference
			size_t idx = &chunk - chunks.data();

			size_t z = idx / (numChunks.x * numChunks.y);
			size_t y = (idx / numChunks.x) % numChunks.y;
			size_t x = idx % numChunks.x;

			chunk.chunkIndex = glm::uvec3(x, y, z);
			extractChunkData(volumeData, idx, chunk);
	});

	// Construct the interval tree
	std::vector<VolumeChunk*> chunkAddresses(numChunksFlat);
	std::for_each(std::execution::par, chunkAddresses.begin(), chunkAddresses.end(),[&](VolumeChunk*& chunkPtr){
		size_t idx = &chunkPtr - chunkAddresses.data(); // compute index
		chunkPtr = &chunks[idx];
	});

	intervalTree.build(chunkAddresses);
}

std::vector<VolumeChunk*> ChunkedVolumeData::query(float isoValue) const
{
	return intervalTree.query(isoValue);
}

void ChunkedVolumeData::destroyStagingBuffer(VulkanEngine* engine)
{
	engine->destroyBuffer(chunksStagingBuffer);
	chunksStagingBuffer.buffer = VK_NULL_HANDLE;
}

glm::uvec3 ChunkedVolumeData::getNumChunks() const
{
	return numChunks;
}

glm::uvec3 ChunkedVolumeData::getChunkSize() const
{
	return chunkSize;
}

size_t ChunkedVolumeData::getNumChunksFlat() const
{
	return chunks.size();
}

VkBuffer ChunkedVolumeData::getStagingBuffer() const
{
	return chunksStagingBuffer.buffer;
}

void* ChunkedVolumeData::getStagingBufferBaseAddress() const
{
	return pChunksStagingBuffer;
}

size_t ChunkedVolumeData::getTotalNumPointsPerChunk() const
{
	return (chunkSize.x + 2) * (chunkSize.y + 2) * (chunkSize.z + 2);
}

glm::uvec3 ChunkedVolumeData::getShellSize() const
{
	return chunkSize + 2u;
}

const std::vector<VolumeChunk>& ChunkedVolumeData::getChunks() const
{
	return chunks;
}

ChunkedVolumeData::~ChunkedVolumeData()
{
	if(chunksStagingBuffer.buffer != VK_NULL_HANDLE)
	{
		pEngine->destroyBuffer(chunksStagingBuffer);
	}
}

/*
	Extracts and writes the data of the chunk into the staging buffer
*/
void ChunkedVolumeData::extractChunkData(const std::vector<uint8_t>& volumeData, size_t chunkFlatIndex, VolumeChunk& chunk)
{
	chunk.chunkFlatIndex = chunkFlatIndex;

	// Compute the bound indices of the chunk for fetching the data from volume data. (Mapping from chunk index to actual grid index)
	glm::uvec3 startIndex = chunkSize * chunk.chunkIndex;
	glm::uvec3 endIndex = glm::min(startIndex + chunkSize + 2u, gridSize); // exclusive end
	
	// TODO: Check this after fixing the thing above
	// Compute the lower and upper corner positions of the chunk from the grid corners.
	glm::vec3 stepSize = (gridUpperCornerPos - gridLowerCornerPos) / glm::vec3(gridSize - 1u);
	chunk.lowerCornerPos = gridLowerCornerPos + glm::vec3(startIndex) * stepSize;
	chunk.upperCornerPos = chunk.lowerCornerPos + glm::vec3(chunkSize) * stepSize;

	chunk.minIsoValue = FLT_MAX;
	chunk.maxIsoValue = -FLT_MAX;

	// Compute and store the offset of the given chunk in the staging buffer
	size_t chunkVoxelsFlatIndex = chunkFlatIndex * ((chunkSize.x + 2) * (chunkSize.y + 2) * (chunkSize.z + 2)); // the actual starting index of the voxels of the chunk in the staging buffer
	chunk.stagingBufferOffset = chunkVoxelsFlatIndex * sizeof(uint8_t);
	
	uint8_t* pChunkVoxels = pChunksStagingBuffer + chunkVoxelsFlatIndex;

	glm::uvec3 pointsPerChunk = chunkSize + 2u; 

	for(size_t z = startIndex.z; z < endIndex.z; ++z)
	{
		for(size_t y = startIndex.y; y < endIndex.y; ++y)
		{
			for(size_t x = startIndex.x; x < endIndex.x; ++x)
			{
				uint8_t val = volumeData[x + gridSize.x * (y + gridSize.y * z)];
				float decompressedVal = val / 255.0f;
				chunk.minIsoValue = std::min(chunk.minIsoValue, decompressedVal);
				chunk.maxIsoValue = std::max(chunk.maxIsoValue, decompressedVal);
				// Write the value into the correct position in the staging buffer
				glm::uvec3 localIdx = glm::uvec3(x, y, z) - startIndex;
				pChunkVoxels[localIdx.x + pointsPerChunk.x * (localIdx.y + pointsPerChunk.y * localIdx.z)] = val;
			}
		}
	}
}
