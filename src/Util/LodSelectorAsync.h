#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdint>

#include "Data/SVO.h"
#include <glm/glm.hpp>

class LODSelectorAsync
{
public:
    struct Params
    {
        glm::vec3 cameraPos;
        float fovY;
        float aspect;
        uint32_t screenHeight;
        float pixelThreshold;
    };

    // ctor: reference to SVO (must outlive this selector)
    // workerThreads: number of background threads (1 is a good default, in my tests multithreads made the performance worse)
    // maxNodesPerTick: budget processed before tiny yields (for smoothnesss)
    // throttleMillis: tiny sleep between chunks (reduce CPU spikes)
    // movementThreshold: world-space movement to trigger recompute
    // minMsBetweenUpdates: minimal time (ms) to force recompute even without big move
    LODSelectorAsync(const SVO& svoRef,
        uint32_t workerThreads = 1,
        size_t maxNodesPerTick = 8000,
        uint32_t throttleMillis = 1,
        float movementThreshold = 0.01f,
        uint32_t minMsBetweenUpdates = 50);

    ~LODSelectorAsync();

    // start/stop worker threads
    void start();
    void stop();

    // Ask the worker to compute selection for the given camera params.
    // Non-blocking. If force==true worker will accept regardless of movement hysteresis.
    void requestUpdate(const Params& p, bool force = false);

    // Copy the latest completed selection into 'out'. Returns number of indices copied.
    size_t getSelectionSnapshot(std::vector<uint32_t>& out) const;

    // Simple tuning accessors
    void setMaxNodesPerTick(size_t v);
    void setThrottleMillis(uint32_t ms);
    void setLODParams(const Params& p);
private:
    LODSelectorAsync(const LODSelectorAsync&) = delete;
    LODSelectorAsync& operator=(const LODSelectorAsync&) = delete;

    // Internal state
    const SVO& svo;

    uint32_t workerCount;
    size_t maxNodesPerTick;
    uint32_t throttleMillis;

    float movementThresholdSq;
    uint32_t minMsBetweenUpdates;

    // Double buffering
    mutable std::vector<uint32_t> buffers[2];
    std::atomic<int> readyBufferIndex;

    std::vector<std::thread> workers;

    mutable std::mutex paramM;
    std::condition_variable cv;
    Params currentParams;
    Params lastParams;
    bool paramsUpdated;
    long long lastUpdateTimeMs;

    std::atomic<bool> stopFlag;
    std::atomic<uint64_t> paramsVersion;
    std::atomic<uint64_t> lastProcessedVersion;
    uint64_t lastAppliedVersion;
};
