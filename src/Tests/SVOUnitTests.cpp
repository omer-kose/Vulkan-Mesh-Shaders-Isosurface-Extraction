#include "SVOUnitTests.h"

void SVOUnitTests::printTestResult(const std::string& testName, bool passed)
{
    fmt::print("{}: {}\n", testName, passed ? "PASSED" : "FAILED");
}

void SVOUnitTests::testBrickedCorrectness()
{
    fmt::print("\n=== Bricked SVO Correctness Test ===\n");

    const int size = 32;
    std::vector<uint8_t> grid(size * size * size, 0);

    // Place voxels in specific corners and edges
    grid[0] = 1; // (0,0,0)
    grid[size - 1] = 2; // (31,0,0)
    grid[(size - 1) + (size - 1) * size] = 3; // (31,31,0)
    grid[(size - 1) + (size - 1) * size + (size - 1) * size * size] = 4; // (31,31,31)

    glm::uvec3 gridSize(size, size, size);
    glm::vec3 worldLower(0.0f), worldUpper(static_cast<float>(size));

    SVO svo(grid, gridSize, worldLower, worldUpper);

    const auto& gpuNodes = svo.getFlatGPUNodes();

    // 1) Basic bounds sanity
    for(const auto& node : gpuNodes) {
        assert(node.lowerCorner.x >= worldLower.x - 1e-6f);
        assert(node.lowerCorner.y >= worldLower.y - 1e-6f);
        assert(node.lowerCorner.z >= worldLower.z - 1e-6f);
        assert(node.upperCorner.x <= worldUpper.x + 1e-6f);
        assert(node.upperCorner.y <= worldUpper.y + 1e-6f);
        assert(node.upperCorner.z <= worldUpper.z + 1e-6f);
    }

    // 2) No overlaps among nodes at the *same level*
    // group nodes by level (NOT by color!)
    std::map<int, std::vector<std::pair<size_t, SVONodeGPU>>> nodesByLevel;
    for(size_t i = 0; i < gpuNodes.size(); ++i) {
        const auto& n = gpuNodes[i];
        nodesByLevel[n.level].push_back({ i, n });
    }

    auto aabbOverlap = [&](const SVONodeGPU& A, const SVONodeGPU& B) -> bool {
        // Use epsilon to avoid FP equality corner cases. Treat touching faces as non-overlap.
        constexpr float eps = 1e-6f;
        bool sepX = (A.upperCorner.x <= B.lowerCorner.x + eps) || (B.upperCorner.x <= A.lowerCorner.x + eps);
        bool sepY = (A.upperCorner.y <= B.lowerCorner.y + eps) || (B.upperCorner.y <= A.lowerCorner.y + eps);
        bool sepZ = (A.upperCorner.z <= B.lowerCorner.z + eps) || (B.upperCorner.z <= A.lowerCorner.z + eps);
        return !(sepX || sepY || sepZ); // overlap if no separating axis
        };

    for(auto& kv : nodesByLevel) {
        int level = kv.first;
        auto& vec = kv.second;
        for(size_t i = 0; i < vec.size(); ++i) {
            for(size_t j = i + 1; j < vec.size(); ++j) {
                const auto& ai = vec[i];
                const auto& bi = vec[j];
                const SVONodeGPU& a = ai.second;
                const SVONodeGPU& b = bi.second;
                if(aabbOverlap(a, b)) {
                    // Helpful debug output before asserting — will show you the two offending nodes
                    fmt::print("Overlap detected at level {} between nodes {} and {}:\n", level, ai.first, bi.first);
                    fmt::print("  A idx={} lvl={} min=({:.6f},{:.6f},{:.6f}) max=({:.6f},{:.6f},{:.6f}) color={}\n",
                        ai.first, a.level, a.lowerCorner.x, a.lowerCorner.y, a.lowerCorner.z,
                        a.upperCorner.x, a.upperCorner.y, a.upperCorner.z, a.colorIndex);
                    fmt::print("  B idx={} lvl={} min=({:.6f},{:.6f},{:.6f}) max=({:.6f},{:.6f},{:.6f}) color={}\n",
                        bi.first, b.level, b.lowerCorner.x, b.lowerCorner.y, b.lowerCorner.z,
                        b.upperCorner.x, b.upperCorner.y, b.upperCorner.z, b.colorIndex);
                    assert(!"Two nodes at same level overlap — investigate coordinates above");
                }
            }
        }
    }

    // 3) LOD sanity: far should not select more nodes than near
    auto nearNodes = svo.selectNodes(glm::vec3(0.0f), 1.0f);
    auto farNodes = svo.selectNodes(glm::vec3(100.0f), 1.0f);
    assert(!nearNodes.empty());
    assert(!farNodes.empty());
    assert(farNodes.size() <= nearNodes.size());

    printTestResult("Bricked SVO Correctness", true);
}

void SVOUnitTests::testBrickedEfficiency()
{
    fmt::print("\n=== Bricked SVO Efficiency Test ===\n");

    std::vector<int> sizes = { 32, 64, 128 };
    for(int size : sizes) {
        glm::uvec3 gridSize(size, size, size);
        glm::vec3 worldLower(0.0f), worldUpper(static_cast<float>(size));

        // Different input patterns
        auto runCase = [&](const std::string& name, auto filler) {
            std::vector<uint8_t> grid(size * size * size, 0);
            for(int z = 0; z < size; z++) {
                for(int y = 0; y < size; y++) {
                    for(int x = 0; x < size; x++) {
                        grid[x + y * size + z * size * size] = filler(x, y, z);
                    }
                }
            }

            auto start = std::chrono::high_resolution_clock::now();
            SVO svo(grid, gridSize, worldLower, worldUpper);
            auto end = std::chrono::high_resolution_clock::now();

            size_t mem = svo.estimateMemoryUsageBytes();
            size_t dense = grid.size() * sizeof(uint8_t);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            fmt::print("[{} {}³] Time={} ms, SVO={} bytes, Dense={} bytes, Ratio={:.2f}x, Nodes={}\n",
                name, size, ms, mem, dense, (double)mem / dense, svo.getFlatGPUNodes().size());
            };

        // Case A: Solid
        runCase("Solid", [&](int, int, int) { return 1; });

        // Case B: Empty
        runCase("Empty", [&](int, int, int) { return 0; });

        // Case C: Sparse random (10%)
        runCase("Sparse10", [&](int, int, int) { return (rand() % 10 == 0) ? 1 : 0; });

        // Case D: Checkerboard
        runCase("Checker", [&](int x, int y, int z) { return ((x + y + z) % 2) ? 1 : 0; });
    }

    printTestResult("Bricked SVO Efficiency", true);
}

void SVOUnitTests::benchmarkLODSimulation()
{
    fmt::print("\n=== LOD Selection Simulation ===\n");

    // Build a demo scene: layered terrain + scattered objects
    const int size = 512; // change to 256/512 based on memory/time
    fmt::print("Scene: {}^3 layered terrain with scattered objects\n", size);

    std::vector<uint8_t> grid(size * size * size, 0);

    // Simple layered terrain + a few pillars (sparse objects)
    for(int x = 0; x < size; ++x) {
        for(int z = 0; z < size; ++z) {
            float nx = float(x) / float(size);
            float nz = float(z) / float(size);
            float height = 0.4f + 0.2f * std::sin(nx * 10.0f) + 0.15f * std::cos(nz * 12.0f);
            int h = std::min(size - 1, int(height * size * 0.5f));
            for(int y = 0; y <= h; ++y) {
                uint8_t mat = (y == h) ? 2u : 1u; // grass top vs stone
                grid[x + y * size + z * size * size] = mat;
            }
        }
    }

    // Add some sparse pillars for detail
    for(int k = 0; k < 300; ++k) {
        int cx = (k * 37) % size;
        int cz = (k * 91) % size;
        int radius = 2 + (k % 4);
        int top = std::min(size - 1, 60 + (k % 20));
        for(int y = 0; y < top; ++y) {
            for(int dz = -radius; dz <= radius; ++dz) {
                for(int dx = -radius; dx <= radius; ++dx) {
                    int px = cx + dx, pz = cz + dz;
                    if(px >= 0 && px < size && pz >= 0 && pz < size) {
                        grid[px + y * size + pz * size * size] = 3; // tower material
                    }
                }
            }
        }
    }

    glm::uvec3 gridSize(size, size, size);
    glm::vec3 worldLower(0.0f), worldUpper(static_cast<float>(size));

    fmt::print("Building SVO (bricks)...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    SVO svo(grid, gridSize, worldLower, worldUpper);
    auto t1 = std::chrono::high_resolution_clock::now();
    double buildMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    size_t mem = svo.estimateMemoryUsageBytes();
    const auto& nodes = svo.getFlatGPUNodes();
    const auto& bricks = svo.getBricks();

    fmt::print("Build time: {:.3f} ms, SVO mem: {} bytes, Nodes: {}, Bricks: {}\n",
        buildMs, mem, nodes.size(), bricks.size());

    // Camera path: grid of camera positions over area, at multiple heights -> simulate frames
    const int camGrid = 12;
    std::vector<glm::vec3> cameraPositions;
    for(int iz = 0; iz < camGrid; ++iz) {
        for(int ix = 0; ix < camGrid; ++ix) {
            float fx = (ix + 0.5f) / float(camGrid);
            float fz = (iz + 0.5f) / float(camGrid);
            float wx = fx * size;
            float wz = fz * size;
            cameraPositions.push_back(glm::vec3(wx, size * 0.6f, wz)); // mid-high camera
            cameraPositions.push_back(glm::vec3(wx, size * 0.3f, wz)); // lower
            cameraPositions.push_back(glm::vec3(wx, size * 0.85f, wz)); // high
        }
    }

    // LOD base distances to test (affects selection granularity)
    std::vector<float> lodBases = { 0.5f, 1.0f, 2.0f, 4.0f };

    // For each lodBase, run selection on all camera positions and gather stats
    for(float lodBase : lodBases) {
        uint64_t totalSelected = 0;
        double totalSelectMs = 0.0;
        std::vector<size_t> perFrameCounts;
        std::map<int, uint64_t> levelCounts; // cumulative per-level
        std::vector<double> perFrameTimes;

        for(const auto& cam : cameraPositions) {
            auto tsel0 = std::chrono::high_resolution_clock::now();
            auto selected = svo.selectNodes(cam, lodBase);
            auto tsel1 = std::chrono::high_resolution_clock::now();
            double selMs = std::chrono::duration_cast<std::chrono::microseconds>(tsel1 - tsel0).count() * 1e-3;
            totalSelectMs += selMs;
            perFrameTimes.push_back(selMs);

            totalSelected += selected.size();
            perFrameCounts.push_back(selected.size());

            // accumulate per-level distribution and avg sizes
            for(uint32_t idx : selected) {
                const auto& n = nodes[idx];
                levelCounts[n.level] += 1;
            }
        }

        double avgPerFrame = double(totalSelected) / double(cameraPositions.size());
        double avgSelectMs = totalSelectMs / double(cameraPositions.size());

        // median per-frame time
        std::sort(perFrameTimes.begin(), perFrameTimes.end());
        double medianMs = perFrameTimes[perFrameTimes.size() / 2];

        fmt::print("\n--- LOD base {:.2f} stats ---\n", lodBase);
        fmt::print("frames={} avgSelected={:.2f} avgSelectMs={:.4f} medianMs={:.4f}\n",
            cameraPositions.size(), avgPerFrame, avgSelectMs, medianMs);

        fmt::print("Per-level selection counts (cumulative across frames):\n");
        for(auto it = levelCounts.rbegin(); it != levelCounts.rend(); ++it) { // coarse -> fine
            fmt::print(" level {} : {} hits\n", it->first, it->second);
        }
    }

    fmt::print("\nLOD selection simulation finished.\n");
}

void SVOUnitTests::benchmarkLargeScaleEfficiency()
{
    fmt::print("\n\n=== Large-Scale Efficiency Benchmark ===\n");

    // sizes to try: be careful with 1024 on low-memory machines
    std::vector<int> sizes = { 128, 256, 512, 1024 };

    for(int size : sizes) {
        // safety: estimate dense bytes and skip if exceeds some cap (for CI/machines)
        size_t denseBytes = size_t(size) * size_t(size) * size_t(size);
        if(denseBytes > (size_t)1024 * 1024 * 1024 * 4ULL) { // >4GB raw -> skip by default
            fmt::print("Skipping {}^3 (raw bytes {}). Adjust cap if you want to run this.\n", size, denseBytes);
            continue;
        }

        fmt::print("\n--- Size {}^3 ---\n", size);

        // We will test 3 input patterns:
        //  1) layered terrain (like earlier)
        //  2) sparse scattered voxels (1 per 4096)
        //  3) dense random (50% occupancy)
        auto runCase = [&](const std::string& name, std::function<uint8_t(int, int, int)> filler) {
            fmt::print("Building grid '{}' for {}^3 ...\n", name, size);

            // allocate grid (may be large)
            std::vector<uint8_t> grid;
            try {
                grid.assign(size_t(size) * size_t(size) * size_t(size), 0u);
            }
            catch(std::bad_alloc&) {
                fmt::print("  Allocation failed for {}^3 - skipping\n", size);
                return;
            }

            for(int z = 0; z < size; ++z)
                for(int y = 0; y < size; ++y)
                    for(int x = 0; x < size; ++x) {
                        grid[x + y * size + z * size * size] = filler(x, y, z);
                    }

            glm::uvec3 gridSize(size, size, size);
            glm::vec3 worldLower(0.0f), worldUpper(static_cast<float>(size));

            auto t0 = std::chrono::high_resolution_clock::now();
            SVO svo(grid, gridSize, worldLower, worldUpper);
            auto t1 = std::chrono::high_resolution_clock::now();
            double buildMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

            size_t mem = svo.estimateMemoryUsageBytes();
            size_t denseBytesLocal = grid.size() * sizeof(uint8_t);
            size_t nodes = svo.getFlatGPUNodes().size();
            size_t bricks = svo.getBricks().size();

            fmt::print("  Build: {:.3f} ms, SVO mem: {} bytes, Dense mem: {} bytes, Ratio={:.2f}x, Nodes={}, Bricks={}\n",
                buildMs, mem, denseBytesLocal, double(mem) / double(denseBytesLocal), nodes, bricks);
            };

        // layered terrain
        runCase("terrain", [&](int x, int y, int z)->uint8_t {
            float nx = float(x) / float(size);
            float nz = float(z) / float(size);
            float height = 0.45f + 0.2f * sin(nx * 10.0f) + 0.15f * cos(nz * 12.0f);
            int h = int(height * size * 0.5f);
            if(y <= h) return (y == h) ? 2 : 1;
            return 0;
            });

        // sparse scattered: 1 in 4096 occupied
        runCase("sparse", [&](int x, int y, int z)->uint8_t {
            uint32_t idx = (uint32_t(x * 73856093u) ^ uint32_t(y * 19349663u) ^ uint32_t(z * 83492791u));
            return ((idx & 0xFFFu) == 0u) ? 4u : 0u; // approx 1/4096
            });

        // dense random 50%
        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> coin(0, 1);
        runCase("random50", [&](int x, int y, int z)->uint8_t {
            return coin(rng) ? 5u : 0u;
            });
    }

    fmt::print("\nLarge-Scale Efficiency Benchmark finished.\n");
}

void SVOUnitTests::benchmarkFineLODSelection()
{
    fmt::print("\n=== Fine LOD Selection Pressure Test ===\n");

    // Use a moderately complex terrain (256^3 or 512^3 is enough)
    const int size = 256;
    std::vector<uint8_t> grid(size * size * size, 0);

    // Terrain-like filler
    for(int x = 0; x < size; ++x) {
        for(int z = 0; z < size; ++z) {
            float nx = float(x) / float(size);
            float nz = float(z) / float(size);
            float h = 0.45f + 0.25f * std::sin(nx * 15.0f) + 0.2f * std::cos(nz * 18.0f);
            int height = int(h * size * 0.7f);
            for(int y = 0; y <= height; ++y)
                grid[x + y * size + z * size * size] = 1;
        }
    }

    glm::uvec3 gridSize(size, size, size);
    glm::vec3 worldLower(0.0f), worldUpper(static_cast<float>(size));

    fmt::print("Building SVO...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    SVO svo(grid, gridSize, worldLower, worldUpper);
    auto t1 = std::chrono::high_resolution_clock::now();
    double buildMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    fmt::print("Build time: {:.3f} ms, nodes={}, bricks={}\n",
        buildMs, svo.getFlatGPUNodes().size(), svo.getBricks().size());

    // Define very close-up camera positions near the terrain surface
    std::vector<glm::vec3> cameras;
    for(int i = 0; i < 12; ++i) {
        float fx = float((i * 37) % size) / float(size);
        float fz = float((i * 91) % size) / float(size);
        float wx = fx * size;
        float wz = fz * size;
        float wy = 12.0f; // really close above terrain
        cameras.push_back(glm::vec3(wx, wy, wz));
    }

    // Use an aggressive LOD base so we force traversal deeper
    float lodBase = 1.0f;

    fmt::print("Testing {} close-up cameras with lodBase={}...\n",
        cameras.size(), lodBase);

    const auto& nodes = svo.getFlatGPUNodes();
    const int brickSize = BRICK_SIZE;

    // Accumulators
    uint64_t totalSelected = 0;
    uint64_t totalVoxels = 0;
    std::map<int, uint64_t> levelHits;

    for(const auto& cam : cameras) {
        auto selected = svo.selectNodes(cam, lodBase);
        totalSelected += selected.size();

        for(uint32_t idx : selected) {
            const auto& n = nodes[idx];
            levelHits[n.level]++;
            if(n.level == 3)
            {
                int x =  3;
            }

            // Estimate voxel contribution: brickSize^3 if leaf brick
            if(n.brickIndex >= 0) {
                totalVoxels += brickSize * brickSize * brickSize;
            }
            else {
                // coarser node covers larger region
                uint64_t side = brickSize << n.level;
                totalVoxels += side * side * side;
            }
        }
    }

    double avgNodes = double(totalSelected) / double(cameras.size());
    double avgVoxels = double(totalVoxels) / double(cameras.size());

    fmt::print("\nResults:\n");
    fmt::print("  Avg selected nodes per frame: {:.2f}\n", avgNodes);
    fmt::print("  Avg covered voxels per frame: {:.0f}\n", avgVoxels);

    fmt::print("  Level distribution:\n");
    for(auto it = levelHits.rbegin(); it != levelHits.rend(); ++it) {
        fmt::print("   level {} : {}\n", it->first, it->second);
    }

    fmt::print("Fine LOD Selection Pressure Test finished.\n");
}

void SVOUnitTests::benchmark()
{
    fmt::print("Running SVO tests...\n");

    try 
    {
        testBrickedCorrectness();
        testBrickedEfficiency();
        benchmarkLODSimulation();
        // benchmarkLargeScaleEfficiency();
        benchmarkFineLODSelection();

        fmt::print("\nAll tests passed!\n");
    }
    catch(const std::exception& e) 
    {
        fmt::print(stderr, "Test failed with exception: {}\n", e.what());
        throw;
    }
}