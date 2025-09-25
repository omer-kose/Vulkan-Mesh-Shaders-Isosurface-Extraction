#include "LODSelectorAsync.h"
#include <algorithm>
#include <cmath>
#include <chrono>

LODSelectorAsync::LODSelectorAsync(const SVO& svoRef,
    unsigned workerThreads,
    size_t maxNodesPerTick_,
    unsigned throttleMillis_,
    float movementThreshold_,
    unsigned minMsBetweenUpdates_)
    : svo(svoRef),
    workerCount(workerThreads),
    maxNodesPerTick(maxNodesPerTick_),
    throttleMillis(throttleMillis_),
    movementThresholdSq(movementThreshold_* movementThreshold_),
    minMsBetweenUpdates(minMsBetweenUpdates_),
    readyBufferIndex(0),
    lastParamsAppliedSet(false),
    lastUpdateTimeMs(0),
    stopFlag(false),
    paramsVersion(0),
    lastProcessedVersion(0),
    lastAppliedVersion(0)
{
    buffers[0].reserve(1024);
    buffers[1].reserve(1024);
}

LODSelectorAsync::~LODSelectorAsync()
{
    stop();
}

void LODSelectorAsync::start()
{
    stopFlag.store(false);
    for(unsigned i = 0; i < workerCount; i++)
    {
        workers.emplace_back([this, i]()
            {
                std::vector<uint32_t> localSelection;
                localSelection.reserve(1024);

                while(!stopFlag.load())
                {
                    Params localParams;
                    uint64_t localVersion = 0;

                    // pick up latest parameters
                    {
                        std::unique_lock<std::mutex> lk(paramM);
                        cv.wait(lk, [this]() { return stopFlag.load() || paramsVersion.load() != lastProcessedVersion.load(); });
                        if(stopFlag.load()) return;

                        localParams = currentParams;
                        localVersion = paramsVersion.load();
                    }

                    // do level-based traversal DFS using iterative stack
                    std::vector<int32_t> stack;
                    stack.reserve(1024);
                    localSelection.clear();

                    for(size_t i = 0; i < svo.nodes.size(); i++)
                    {
                        if(svo.nodes[i].parentIndex == -1)
                            stack.push_back((int32_t)i);
                    }

                    size_t processedNodes = 0;
                    float screenFactor = float(localParams.screenHeight) / (2.0f * std::tan(localParams.fovY * 0.5f));

                    while(!stack.empty())
                    {
                        int32_t nodeIdx = stack.back();
                        stack.pop_back();

                        const SVO::Node& n = svo.nodes[nodeIdx];
                        const SVONodeGPU& gn = svo.getFlatGPUNodes()[n.flatIndex];

                        glm::vec3 center = (gn.lowerCorner + gn.upperCorner) * 0.5f;
                        glm::vec3 ext = gn.upperCorner - gn.lowerCorner;
                        float nodeExtent = std::max(std::max(ext.x, ext.y), ext.z);
                        float dist = glm::length(localParams.cameraPos - center);
                        if(dist <= 0.0f) dist = 0.0001f;

                        float screenSize = (nodeExtent * screenFactor) / dist;
                        bool needsRefinement = (screenSize > localParams.pixelThreshold);
                        bool isLeaf = (n.level == svo.getLeafLevel());

                        if(!needsRefinement || isLeaf)
                        {
                            localSelection.push_back(static_cast<uint32_t>(n.flatIndex));
                        }
                        else
                        {
                            for(int i = 0; i < 8; i++)
                            {
                                int32_t c = n.children[i];
                                if(c >= 0) stack.push_back(c);
                            }
                        }

                        processedNodes++;
                        if(processedNodes >= maxNodesPerTick)
                        {
                            processedNodes = 0;
                            std::this_thread::sleep_for(std::chrono::milliseconds(throttleMillis));
                        }
                    }

                    // publish to buffer
                    int nextIndex = 1 - readyBufferIndex.load();
                    {
                        std::unique_lock<std::mutex> lk(paramM);
                        buffers[nextIndex] = std::move(localSelection);
                        lastProcessedVersion.store(localVersion);
                        readyBufferIndex.store(nextIndex);
                    }
                }
            });
    }
}

void LODSelectorAsync::stop()
{
    stopFlag.store(true);
    cv.notify_all();
    for(auto& t : workers)
    {
        if(t.joinable())
            t.join();
    }
    workers.clear();
}

void LODSelectorAsync::requestUpdate(const Params& p, bool force)
{
    std::unique_lock<std::mutex> lk(paramM);

    // compute distance to last applied camera
    bool needUpdate = force;
    if(!force && lastParamsAppliedSet)
    {
        glm::vec3 diff = p.cameraPos - lastParams.cameraPos;
        if(glm::dot(diff, diff) >= movementThresholdSq)
            needUpdate = true;

        if(!needUpdate)
        {
            // check min time between forced updates
            auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            if(nowMs - lastUpdateTimeMs >= (long long)minMsBetweenUpdates)
                needUpdate = true;
        }
    }

    if(needUpdate)
    {
        currentParams = p;
        paramsVersion.fetch_add(1);
        lastParams = p;
        lastParamsAppliedSet = true;
        lastUpdateTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

        cv.notify_all();
    }
}

size_t LODSelectorAsync::getSelectionSnapshot(std::vector<uint32_t>& out) const
{
    int idx = readyBufferIndex.load();
    std::unique_lock<std::mutex> lk(paramM);
    out = buffers[idx];
    return out.size();
}

const std::vector<uint32_t>& LODSelectorAsync::getSelectionRef() const
{
    return buffers[readyBufferIndex.load()];
}

void LODSelectorAsync::setMaxNodesPerTick(size_t v)
{
    maxNodesPerTick = v;
}

void LODSelectorAsync::setThrottleMillis(unsigned ms)
{
    throttleMillis = ms;
}

void LODSelectorAsync::setLODParams(const Params& p)
{
    currentParams = p;
    paramsVersion.fetch_add(1);
    lastParams = p;
    lastParamsAppliedSet = true;
}
