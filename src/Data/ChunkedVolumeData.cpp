#include "ChunkedVolumeData.h"

#include <Core/vk_engine.h>

#include <Pass/MarchingCubesLookup.h>

#include <algorithm>
#include <execution>

ChunkedVolumeData::ChunkedVolumeData(VulkanEngine* engine, const std::vector<float>& volumeData, const glm::uvec3& gridSize_in, const glm::uvec3& chunkSize_in, const glm::vec3& gridLowerCornerPos_in, const glm::vec3& gridUpperCornerPos_in)
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
		pChunksStagingBuffer[i] = 0.0f;
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

size_t ChunkedVolumeData::getNumChunksFlat() const
{
	return chunks.size();
}

VkBuffer ChunkedVolumeData::getStagingBuffer() const
{
	return chunksStagingBuffer.buffer;
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

void ChunkedVolumeData::computeChunkIsoValueHistograms(float minIsoValue, float maxIsoValue, size_t numBins)
{
	float stepSize = (maxIsoValue - minIsoValue) / (numBins - 1);
	glm::uvec3 pointsPerChunk = chunkSize + 2u;

	// Parallelize outer loop over chunks
	std::for_each(std::execution::par, chunks.begin(), chunks.end(), [&](VolumeChunk& c)
	{
		size_t chunkFloatOffset = c.stagingBufferOffset / sizeof(float);
		float* pChunk = pChunksStagingBuffer + chunkFloatOffset;

		// Pre-allocate and initialize the histogram bins
		c.isoValueHistogram.resize(numBins);
		for(size_t b = 0; b < numBins; ++b)
		{
			c.isoValueHistogram[b] = { minIsoValue + b * stepSize, 0 };
		}

		float values[8];

		for(size_t z = 0; z < chunkSize.z; ++z)
		{
			for(size_t y = 0; y < chunkSize.y; ++y)
			{
				for(size_t x = 0; x < chunkSize.x; ++x)
				{
					// Fetch voxel corner values
					values[0] = pChunk[(x + 0) + pointsPerChunk.x * ((y + 0) + pointsPerChunk.y * (z + 0))];
					values[1] = pChunk[(x + 1) + pointsPerChunk.x * ((y + 0) + pointsPerChunk.y * (z + 0))];
					values[2] = pChunk[(x + 0) + pointsPerChunk.x * ((y + 1) + pointsPerChunk.y * (z + 0))];
					values[3] = pChunk[(x + 1) + pointsPerChunk.x * ((y + 1) + pointsPerChunk.y * (z + 0))];
					values[4] = pChunk[(x + 0) + pointsPerChunk.x * ((y + 0) + pointsPerChunk.y * (z + 1))];
					values[5] = pChunk[(x + 1) + pointsPerChunk.x * ((y + 0) + pointsPerChunk.y * (z + 1))];
					values[6] = pChunk[(x + 0) + pointsPerChunk.x * ((y + 1) + pointsPerChunk.y * (z + 1))];
					values[7] = pChunk[(x + 1) + pointsPerChunk.x * ((y + 1) + pointsPerChunk.y * (z + 1))];

					// Loop through bins
					for(size_t b = 0; b < numBins; ++b)
					{
						float currentIsoValue = c.isoValueHistogram[b].first;

						uint32_t cubeIndex = 0;
						cubeIndex |= (values[0] >= currentIsoValue) << 0;
						cubeIndex |= (values[1] >= currentIsoValue) << 1;
						cubeIndex |= (values[2] >= currentIsoValue) << 2;
						cubeIndex |= (values[3] >= currentIsoValue) << 3;
						cubeIndex |= (values[4] >= currentIsoValue) << 4;
						cubeIndex |= (values[5] >= currentIsoValue) << 5;
						cubeIndex |= (values[6] >= currentIsoValue) << 6;
						cubeIndex |= (values[7] >= currentIsoValue) << 7;

						if(cubeIndex != 0 && cubeIndex != 0xFF)
						{
							c.isoValueHistogram[b].second += MarchingCubesLookupTable[cubeIndex].TriangleCount;
						}
					}
				}
			}
		}
	});
}

size_t ChunkedVolumeData::estimateNumTriangles(const VolumeChunk& chunk, float isoValue) const
{
	const std::vector<std::pair<float, size_t>>& h = chunk.isoValueHistogram;
	// Find the left and right bins that input iso value falls between (the histogram itself is sorted so a binary range search is possible)
	int left = 0;
	int right = h.size() - 1;
	while(right - left > 1)
	{
		int mid = (left + right) / 2;
		float iso = h[mid].first;
		if(iso < isoValue)
		{
			left = mid;
		}
		else
		{
			right = mid;
		}
	}

	// Estimate by lerping and rounding
	float alpha = (isoValue - h[left].first) / (h[right].first - h[left].first);
	return (1.0f - alpha) * h[left].second + alpha * h[right].second;
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
	glm::uvec3 endIndex = glm::min(startIndex + chunkSize + 2u, gridSize); // exclusive end
	
	// TODO: Check this after fixing the thing above
	// Compute the lower and upper corner positions of the chunk from the grid corners.
	glm::vec3 stepSize = (gridUpperCornerPos - gridLowerCornerPos) / glm::vec3(gridSize - 1u);
	chunk.lowerCornerPos = gridLowerCornerPos + glm::vec3(startIndex) * stepSize;
	chunk.upperCornerPos = chunk.lowerCornerPos + glm::vec3(chunkSize) * stepSize;

	chunk.minIsoValue = FLT_MAX;
	chunk.maxIsoValue = -FLT_MAX;

	// Compute and store the offset of the given chunk in the staging buffer
	size_t chunkVoxelsFlatIndex = flatChunkIndex * ((chunkSize.x + 2) * (chunkSize.y + 2) * (chunkSize.z + 2)); // the actual starting index of the voxels of the chunk in the staging buffer
	chunk.stagingBufferOffset = chunkVoxelsFlatIndex * sizeof(float);
	
	float* pChunkVoxels = pChunksStagingBuffer + chunkVoxelsFlatIndex;

	glm::uvec3 pointsPerChunk = chunkSize + 2u; 

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
				pChunkVoxels[localIdx.x + pointsPerChunk.x * (localIdx.y + pointsPerChunk.y * localIdx.z)] = val;
			}
		}
	}
}
