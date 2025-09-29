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

void SVOUnitTests::testBrickedCorrectnessNonPowerOfTwo()
{
    fmt::print("\n=== Bricked SVO Correctness Test ===\n");

    const int size = 31;
    std::vector<uint8_t> grid(size * size * size, 0);

    // Place voxels in specific corners and edges
    grid[0] = 1; // (0,0,0)
    grid[size - 1] = 2; // (30,0,0)
    grid[(size - 1) + (size - 1) * size] = 3; // (30,30,0)
    grid[(size - 1) + (size - 1) * size + (size - 1) * size * size] = 4; // (30,30,30)

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

    printTestResult("Bricked SVO Correctness Non Power of Two", true);
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
    const auto& bricks = svo.getFineBricks();

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
            size_t bricks = svo.getFineBricks().size();

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
    const int size = 512;
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
        buildMs, svo.getFlatGPUNodes().size(), svo.getFineBricks().size());

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
    const int brickSize = FINE_BRICK_SIZE;

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

void SVOUnitTests::testLargeWorldScreenSpaceLOD()
{
    // ===== Generate a large sparse world =====
// Grid: SIZE^3 voxels
    constexpr int SIZE = 512; // node grid size (small enough to handle memory)
    std::vector<uint8_t> grid(SIZE * SIZE * SIZE, 0);

    // World size (massive for meaningful LOD)
    const float worldScale = 8192.0f; // 8 km world, for example
    glm::vec3 worldLower(0);
    glm::vec3 worldUpper(worldScale);

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> heightDist(50, 300);

    // Ground / mountain range
    for(int x = 0; x < SIZE; ++x) {
        for(int z = 0; z < SIZE; ++z) {
            int h = heightDist(rng);
            for(int y = 0; y <= h; ++y)
                grid[x + y * SIZE + z * SIZE * SIZE] = 1;
        }
    }

    // Clouds
    const int cloudStartY = 400;
    const int cloudThickness = 32;
    const uint8_t cloudColor = 2;
    for(int x = 50; x < SIZE - 50; x += 32) {
        for(int z = 50; z < SIZE - 50; z += 32) {
            for(int y = cloudStartY; y < cloudStartY + cloudThickness; ++y) {
                for(int dx = 0; dx < 16; ++dx)
                    for(int dz = 0; dz < 16; ++dz)
                        grid[(x + dx) + y * SIZE + (z + dz) * SIZE * SIZE] = cloudColor;
            }
        }
    }

    glm::uvec3 gridSize(SIZE, SIZE, SIZE);

    // ===== Build SVO without collapsing =====
    fmt::print("Building SVO (brick compaction only, no node collapsing)...\n");
    SVO svo(grid, gridSize, worldLower, worldUpper);
    auto memUsage = svo.estimateMemoryUsageBytes();
    fmt::print("SVO built: nodes={}, bricks={}, mem={} bytes\n",
        svo.nodes.size(), svo.fineBricks.size(), memUsage);

    // ===== LOD Selection Benchmark =====
    std::vector<glm::vec3> cameras = {
        {worldScale / 2.0f, worldScale / 2.0f, worldScale * 2.0f}, // overhead
        {0.0f, 150.0f, 0.0f},      // corner
        {worldScale, 150.0f, worldScale}, // opposite corner
        {worldScale / 2.0f, 150.0f, 0.0f}     // mid-edge
    };

    float pixelThresholds[] = { 1.0f, 4.0f, 16.0f, 32.0f };

    for(float pixThr : pixelThresholds) {
        fmt::print("\n--- LOD Selection Benchmark (pixelThreshold = {}) ---\n", pixThr);

        for(size_t c = 0; c < cameras.size(); ++c) {
            auto camPos = cameras[c];

            auto selected = svo.selectNodesScreenSpace(camPos, glm::radians(45.0f), 16.0f / 9.0f, 1080, pixThr);

            // Histogram by node.level
            std::map<uint8_t, size_t> levelHist;
            for(auto idx : selected) {
                levelHist[svo.nodes[idx].level]++;
            }

            fmt::print("Camera {} -> selected {} nodes\n", c, selected.size());
            for(auto [lvl, cnt] : levelHist) {
                fmt::print("   level {:2d} : {:6d} ({:6.2f}%)\n", lvl, (int)cnt, 100.0 * cnt / selected.size());
            }
        }
    }
}

//void SVOUnitTests::testVoxelCoverage()
//{
//    const int size = 512;
//    std::vector<uint8_t> grid(size * size * size, 0);
//
//    // Terrain-like filler
//    for(int x = 0; x < size; ++x) {
//        for(int z = 0; z < size; ++z) {
//            float nx = float(x) / float(size);
//            float nz = float(z) / float(size);
//            float h = 0.45f + 0.25f * std::sin(nx * 15.0f) + 0.2f * std::cos(nz * 18.0f);
//            int height = int(h * size * 0.7f);
//            for(int y = 0; y <= height; ++y)
//                grid[x + y * size + z * size * size] = 1;
//        }
//    }
//
//    glm::uvec3 gridSize(size, size, size);
//    glm::vec3 worldLower(0.0f), worldUpper(static_cast<float>(size));
//
//    // Build two SVOs
//    SVO svoCollapsed(grid, gridSize, worldLower, worldUpper, /*collapseNodes=*/true);
//    SVO svoUncollapsed(grid, gridSize, worldLower, worldUpper, /*collapseNodes=*/false);
//
//    auto coverVoxels = [&](const SVO& svo) {
//        std::vector<uint8_t> mask(size * size * size, 0);
//
//        auto markNode = [&](const SVO::Node& n) {
//            glm::vec3 mn, mx;
//            svo.computeWorldAABB(n, mn, mx);
//
//            glm::ivec3 imin = glm::floor((mn - worldLower) / svo.voxelSize);
//            glm::ivec3 imax = glm::ceil((mx - worldLower) / svo.voxelSize);
//
//            imin = glm::clamp(imin, glm::ivec3(0), glm::ivec3(size - 1));
//            imax = glm::clamp(imax, glm::ivec3(0), glm::ivec3(size - 1));
//
//            for(int z = imin.z; z <= imax.z; ++z)
//                for(int y = imin.y; y <= imax.y; ++y)
//                    for(int x = imin.x; x <= imax.x; ++x) {
//                        uint32_t idx = x + y * size + z * size * size;
//                        mask[idx] = 1;
//                    }
//            };
//
//        for(const auto& n : svo.nodes) {
//            if(n.childrenMask == 0) { // leaf or collapsed node
//                markNode(n);
//            }
//        }
//
//        return mask;
//        };
//
//    auto mCollapsed = coverVoxels(svoCollapsed);
//    auto mUncollapsed = coverVoxels(svoUncollapsed);
//
//    assert(mCollapsed.size() == mUncollapsed.size());
//
//    size_t mismatches = 0;
//    std::unordered_map<int, size_t> mismatchesPerLevel;
//
//    for(size_t i = 0; i < mCollapsed.size(); ++i) {
//        if(mCollapsed[i] != mUncollapsed[i]) {
//            ++mismatches;
//
//            // Find which node in uncollapsed covered this voxel
//            glm::uvec3 v(
//                i % size,
//                (i / size) % size,
//                i / (size * size)
//            );
//
//            for(const auto& n : svoUncollapsed.nodes) {
//                if(n.childrenMask == 0) {
//                    glm::vec3 mn, mx;
//                    svoUncollapsed.computeWorldAABB(n, mn, mx);
//                    glm::vec3 p = glm::vec3(v) * svoUncollapsed.voxelSize + worldLower;
//                    if(p.x >= mn.x && p.y >= mn.y && p.z >= mn.z &&
//                        p.x < mx.x && p.y < mx.y && p.z < mx.z) {
//                        mismatchesPerLevel[n.level]++;
//                        break;
//                    }
//                }
//            }
//        }
//    }
//
//    fmt::print("=== Voxel Coverage Test (terrain  << {} << ^3) ===\n", size);
//    if(mismatches == 0) {
//        fmt::print("  PASSED: no mismatches between collapsed and uncollapsed\n");
//    }
//    else {
//        fmt::print("  FAILED:  << {} <<  mismatches\n", mismatches);
//        for(auto& kv : mismatchesPerLevel) {
//            fmt::print("   Level  << {} <<  :  << {} <<  mismatches\n", kv.first, kv.second);
//        }
//    }
//}
//
//void SVOUnitTests::testScreenSpaceLOD()
//{
//    fmt::print("=== Screen-Space LOD Benchmark ===\n");
//
//    // Build terrain-like grid (512^3)
//    const int size = 512;
//    std::vector<uint8_t> grid(size * size * size, 0);
//
//    for(int x = 0; x < size; ++x) {
//        for(int z = 0; z < size; ++z) {
//            float nx = float(x) / float(size);
//            float nz = float(z) / float(size);
//            float h = 0.45f + 0.25f * std::sin(nx * 15.0f) + 0.2f * std::cos(nz * 18.0f);
//            int height = int(h * size * 0.7f);
//            for(int y = 0; y <= height; ++y)
//                grid[x + y * size + z * size * size] = 1;
//        }
//    }
//
//    glm::uvec3 gridSize(size, size, size);
//    glm::vec3 worldLower(0.0f);
//    glm::vec3 worldUpper(size);
//
//    // Build SVO
//    auto start = std::chrono::high_resolution_clock::now();
//    SVO svo(grid, gridSize, worldLower, worldUpper, true); // collapse = true
//    auto end = std::chrono::high_resolution_clock::now();
//    double buildMs = std::chrono::duration<double, std::milli>(end - start).count();
//
//    fmt::print("Built SVO in {:.3f} ms, nodes={}, bricks={}\n",
//        buildMs, svo.nodes.size(), svo.bricks.size());
//
//    // Camera setup
//    float fovY = glm::radians(60.0f);
//    float aspect = 16.0f / 9.0f;
//    uint32_t screenHeight = 1080;
//    float pixelThreshold = 2.0f;
//
//    std::vector<glm::vec3> cameraPositions = {
//        glm::vec3(size * 0.5f, size * 1.2f, size * 0.5f),   // far above
//        glm::vec3(size * 0.5f, size * 0.6f, -size * 0.2f),  // medium distance
//        glm::vec3(size * 0.5f, size * 0.4f, size * 0.5f + 20) // close-up
//    };
//
//    for(size_t i = 0; i < cameraPositions.size(); ++i) {
//        auto cam = cameraPositions[i];
//
//        auto startSel = std::chrono::high_resolution_clock::now();
//        auto selected = svo.selectNodesScreenSpace(
//            cam, fovY, aspect, screenHeight, pixelThreshold);
//        auto endSel = std::chrono::high_resolution_clock::now();
//        double ms = std::chrono::duration<double, std::milli>(endSel - startSel).count();
//
//        fmt::print("Camera {} at ({:.1f},{:.1f},{:.1f}) "
//            "-> selected {} nodes in {:.3f} ms\n",
//            i, cam.x, cam.y, cam.z, selected.size(), ms);
//
//        // Debug root values
//        const auto& root = svo.nodes[0];
//        glm::vec3 minW, maxW;
//        svo.computeWorldAABB(root, minW, maxW);
//        float dist = svo.distanceToAABB(cam, minW, maxW);
//        float nodeSize = maxW.x - minW.x;
//        float projectedSize = (nodeSize / std::max(dist, 1e-3f)) * (float(screenHeight) / (2.0f * std::tan(fovY * 0.5f)));
//
//        fmt::print("   Root: nodeSize={:.2f}, dist={:.2f}, projectedSize={:.2f}, childrenMask={:02X}\n",
//            nodeSize, dist, projectedSize, root.childrenMask);
//    }
//
//    fmt::print("Screen-Space LOD Benchmark finished.\n");
//}
//
//void SVOUnitTests::testScreenSpaceLODHistogram()
//{
//    fmt::println("=== Screen-Space LOD Histogram Benchmark ===");
//
//    const int chunkSize = 64;
//    const int chunks = 8; // world = 1024
//    const int size = chunkSize * chunks;
//    std::vector<uint8_t> grid(size * size * size, 0);
//
//    for(int cx = 0; cx < chunks; ++cx) {
//        for(int cz = 0; cz < chunks; ++cz) {
//            for(int x = 0; x < chunkSize; ++x) {
//                for(int z = 0; z < chunkSize; ++z) {
//                    int gx = cx * chunkSize + x;
//                    int gz = cz * chunkSize + z;
//
//                    float nx = float(x) / float(chunkSize);
//                    float nz = float(z) / float(chunkSize);
//                    float h = 0.45f + 0.25f * std::sin(nx * 15.0f) + 0.2f * std::cos(nz * 18.0f);
//                    int height = int(h * chunkSize * 0.7f);
//
//                    for(int y = 0; y <= height; ++y)
//                        grid[gx + y * size + gz * size * size] = 1;
//                }
//            }
//        }
//    }
//
//    glm::uvec3 gridSize(size, size, size);
//    glm::vec3 worldLower(0.0f);
//    glm::vec3 worldUpper(size);
//
//    // Build two SVOs: no collapse and collapse
//    fmt::println("Building SVO (collapse = false) ...");
//    auto t0 = std::chrono::high_resolution_clock::now();
//    SVO svoNoCollapse(grid, gridSize, worldLower, worldUpper, false);
//    auto t1 = std::chrono::high_resolution_clock::now();
//    double buildMsNo = std::chrono::duration<double, std::milli>(t1 - t0).count();
//
//    fmt::println("Building SVO (collapse = true) ...");
//    auto t2 = std::chrono::high_resolution_clock::now();
//    SVO svoCollapse(grid, gridSize, worldLower, worldUpper, true);
//    auto t3 = std::chrono::high_resolution_clock::now();
//    double buildMsYes = std::chrono::duration<double, std::milli>(t3 - t2).count();
//
//    fmt::println("Built (no-collapse)  in {:.3f} ms, nodes={}, bricks={}", buildMsNo, svoNoCollapse.nodes.size(), svoNoCollapse.bricks.size());
//    fmt::println("Built (collapsed)   in {:.3f} ms, nodes={}, bricks={}", buildMsYes, svoCollapse.nodes.size(), svoCollapse.bricks.size());
//
//    // Camera positions to probe (far -> near)
//    std::vector<glm::vec3> cameras = {
//        glm::vec3(size * 0.5f, size * 0.5f, size * 20.0f),   // 2x world away
//        glm::vec3(size * 0.5f, size * 20.0f, size * 0.5f),    // far above
//        glm::vec3(-size * 30.0f, size * 0.5f, size * 0.5f)   // very far side
//    };
//
//    float fovY = glm::radians(45.0f);
//    float aspect = 16.0f / 9.0f;
//    uint32_t screenH = 1080;
//    float pixelThreshold = 2.0f;
//
//    auto runForSVO = [&](const SVO& svo, const char* name) {
//        fmt::println("\n--- Results for: {} ---", name);
//
//        for(size_t ci = 0; ci < cameras.size(); ++ci) {
//            const glm::vec3& cam = cameras[ci];
//
//            auto s_st = std::chrono::high_resolution_clock::now();
//            auto selected = svo.selectNodesScreenSpace(cam, fovY, aspect, screenH, pixelThreshold);
//            auto s_en = std::chrono::high_resolution_clock::now();
//            double selMs = std::chrono::duration<double, std::milli>(s_en - s_st).count();
//
//            // histogram per level
//            std::vector<size_t> hist(svo.levels, 0);
//            for(uint32_t nid : selected) {
//                uint8_t lvl = svo.nodes[nid].level;
//                if(lvl < hist.size()) hist[lvl]++;
//            }
//
//            fmt::println("Camera {} at ({:.1f},{:.1f},{:.1f}) -> selected {} nodes in {:.3f} ms",
//                ci, cam.x, cam.y, cam.z, selected.size(), selMs);
//
//            // print top-down (coarsest -> finest)
//            bool any = false;
//            for(int L = svo.levels - 1; L >= 0; --L) {
//                if(hist[L] > 0) {
//                    fmt::print("   level {:2} : {}", L, hist[L]);
//                    // percentage of total selected at that level
//                    double pct = double(hist[L]) / double(std::max<size_t>(1, selected.size())) * 100.0;
//                    fmt::println("  ({:.2f}%)", pct);
//                    any = true;
//                }
//            }
//            if(!any) fmt::println("   level distribution: (none)");
//
//            // quick summary: finest-level count (leafLevel) and coarsest count
//            size_t finestCount = (svo.leafLevel < hist.size()) ? hist[svo.leafLevel] : 0;
//            size_t coarsestCount = (svo.levels > 0) ? hist[svo.levels - 1] : 0;
//            fmt::println("   summary: finest(level={}) = {}, coarsest(level={}) = {}",
//                svo.leafLevel, finestCount, svo.levels - 1, coarsestCount);
//        }
//        };
//
//    // Run both
//    runForSVO(svoNoCollapse, "SVO (collapse = false)");
//    runForSVO(svoCollapse, "SVO (collapse = true)");
//
//    // Comparative summary per camera
//    fmt::println("\n--- Comparative summary (selected node counts per camera) ---");
//    for(size_t ci = 0; ci < cameras.size(); ++ci) {
//        auto selA = svoNoCollapse.selectNodesScreenSpace(cameras[ci], fovY, aspect, screenH, pixelThreshold);
//        auto selB = svoCollapse.selectNodesScreenSpace(cameras[ci], fovY, aspect, screenH, pixelThreshold);
//        fmt::println("Camera {} : no-collapse = {}, collapsed = {}, ratio(collapsed/no) = {:.3f}",
//            ci, selA.size(), selB.size(), (selA.size() == 0 ? 0.0 : double(selB.size()) / double(selA.size())));
//    }
//
//    fmt::println("\nScreen-Space LOD Histogram Benchmark finished.");
//}
//
//void SVOUnitTests::testCompactionMemorySavings()
//{
//    fmt::println("\n=== SVO Compaction Memory Test ===");
//
//    // smallish grid to keep time reasonable - you can bump this
//    const int size = 256;
//    std::vector<uint8_t> grid(size * size * size, 0);
//
//    // simple filled terrain (dense-ish) so collapse can fire
//    for(int x = 0; x < size; ++x) {
//        for(int z = 0; z < size; ++z) {
//            float nx = float(x) / float(size);
//            float nz = float(z) / float(size);
//            float h = 0.45f + 0.25f * std::sin(nx * 6.0f) + 0.2f * std::cos(nz * 8.0f);
//            int height = int(h * size * 0.7f);
//            for(int y = 0; y <= height; ++y) grid[x + y * size + z * size * size] = (y % 2 == 0 ? 1u : 2u);
//        }
//    }
//
//    glm::uvec3 gsize(size, size, size);
//    glm::vec3 wl(0.0f), wu(size);
//
//    // Build a collapsed SVO
//    auto t0 = std::chrono::high_resolution_clock::now();
//    SVO svoCollapsed(grid, gsize, wl, wu, true);
//    auto t1 = std::chrono::high_resolution_clock::now();
//    double buildMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
//
//    size_t beforeBytes = svoCollapsed.estimateMemoryUsageBytes();
//    size_t beforeNodes = svoCollapsed.nodes.size();
//    size_t beforeBricks = svoCollapsed.bricks.size();
//
//    fmt::println("Built collapsed SVO: nodes={}, bricks={}, mem={} bytes (build {:.3f} ms)",
//        beforeNodes, beforeBricks, beforeBytes, buildMs);
//
//    // Compact it
//    auto t2 = std::chrono::high_resolution_clock::now();
//    svoCollapsed.compact();
//    auto t3 = std::chrono::high_resolution_clock::now();
//    double compactMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
//
//    size_t afterBytes = svoCollapsed.estimateMemoryUsageBytes();
//    size_t afterNodes = svoCollapsed.nodes.size();
//    size_t afterBricks = svoCollapsed.bricks.size();
//
//    fmt::println("After compact: nodes={}, bricks={}, mem={} bytes (compact {:.3f} ms)",
//        afterNodes, afterBricks, afterBytes, compactMs);
//
//    // Build non-collapsed SVO for comparison
//    SVO svoNoCollapse(grid, gsize, wl, wu, false);
//    size_t noCollapseBytes = svoNoCollapse.estimateMemoryUsageBytes();
//    size_t noCollapseNodes = svoNoCollapse.nodes.size();
//    size_t noCollapseBricks = svoNoCollapse.bricks.size();
//
//    fmt::println("Non-collapsed SVO: nodes={}, bricks={}, mem={} bytes",
//        noCollapseNodes, noCollapseBricks, noCollapseBytes);
//
//    // Sanity checks: compaction should not increase memory
//    assert(afterBytes <= beforeBytes && "Compact should not increase memory");
//    // and collapsed should not be larger than non-collapsed after compaction (ideally smaller)
//    assert(afterBytes <= noCollapseBytes + (size_t)1e6 && "Collapsed should be no larger than non-collapsed (allow small noise)");
//    // For reporting
//    fmt::println("Memory saved: beforeCompact={} bytes -> afterCompact={} bytes (ratio {:.2f}x)",
//        beforeBytes, afterBytes, (double)beforeBytes / double(afterBytes + 1u));
//
//    fmt::println("SVO Compaction Memory Test: PASSED");
//}
//
//void SVOUnitTests::runComprehensiveSVOTest()
//{
//    fmt::println("\n=== Comprehensive SVO Test ===");
//
//    // Build a tiled world: tileCount * baseChunk
//    const int baseChunk = 128; // base procedural chunk size
//    const int tileCount = 4;   // world size = baseChunk * tileCount (512 here)
//    const int size = baseChunk * tileCount;
//
//    fmt::println("Generating tiled world {}^3 (base {} * tiles {}) ...", size, baseChunk, tileCount);
//
//    std::vector<uint8_t> grid(size * size * size, 0);
//    for(int tx = 0; tx < tileCount; ++tx) {
//        for(int tz = 0; tz < tileCount; ++tz) {
//            // place a procedural hill per tile to create variety
//            for(int x = 0; x < baseChunk; ++x) {
//                for(int z = 0; z < baseChunk; ++z) {
//                    int gx = tx * baseChunk + x;
//                    int gz = tz * baseChunk + z;
//                    float nx = float(x) / float(baseChunk);
//                    float nz = float(z) / float(baseChunk);
//                    float h = 0.45f + 0.25f * std::sin(nx * 6.0f * (tx + 1)) + 0.2f * std::cos(nz * 8.0f * (tz + 1));
//                    int height = int(h * baseChunk * 0.6f);
//                    for(int y = 0; y <= height; ++y) {
//                        grid[gx + y * size + gz * size * size] = (uint8_t)((y % 3) + 1); // some palette variety
//                    }
//                }
//            }
//        }
//    }
//
//    glm::uvec3 gsize(size, size, size);
//    glm::vec3 wl(0.0f), wu(size);
//
//    // Build non-collapsed
//    auto t0 = std::chrono::high_resolution_clock::now();
//    SVO svoNo(grid, gsize, wl, wu, false);
//    auto t1 = std::chrono::high_resolution_clock::now();
//    double buildNoMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
//
//    // Build collapsed & compacted
//    auto t2 = std::chrono::high_resolution_clock::now();
//    SVO svoCollapsed(grid, gsize, wl, wu, true);
//    auto t3 = std::chrono::high_resolution_clock::now();
//    double buildCollapsedMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
//
//    // Compact collapsed SVO
//    auto t4 = std::chrono::high_resolution_clock::now();
//    svoCollapsed.compact();
//    auto t5 = std::chrono::high_resolution_clock::now();
//    double compactMs = std::chrono::duration<double, std::milli>(t5 - t4).count();
//
//    fmt::println("Built NON-COLLAPSED in {:.3f} ms, nodes={}, bricks={}, mem={}", buildNoMs, svoNo.nodes.size(), svoNo.bricks.size(), svoNo.estimateMemoryUsageBytes());
//    fmt::println("Built COLLAPSED  in {:.3f} ms, nodes={}, bricks={}, mem={} (compacted in {:.3f} ms)", buildCollapsedMs, svoCollapsed.nodes.size(), svoCollapsed.bricks.size(), svoCollapsed.estimateMemoryUsageBytes(), compactMs);
//
//    // LOD selection probe settings
//    std::vector<glm::vec3> cameras = {
//        glm::vec3(size * 0.5f, size * 0.5f, -size * 5.0f),   // very far
//        glm::vec3(size * 0.5f, size * 1.5f, size * 0.5f),    // far above
//        glm::vec3(size * 0.5f, size * 0.7f, size * 0.5f + 50), // medium
//        glm::vec3(size * 0.5f, size * 0.45f, size * 0.5f + 10) // close
//    };
//
//    std::vector<float> pixelThresholds = { 1.0f, 2.0f, 8.0f, 16.0f };
//
//    float fovY = glm::radians(60.0f);
//    float aspect = 16.0f / 9.0f;
//    uint32_t screenH = 1080;
//
//    auto runProbe = [&](const SVO& svo, const char* name) {
//        fmt::println("\n--- LOD probes for {} ---", name);
//
//        for(size_t ci = 0; ci < cameras.size(); ++ci) {
//            const auto& cam = cameras[ci];
//            for(float thr : pixelThresholds) {
//                auto st = std::chrono::high_resolution_clock::now();
//                auto sel = svo.selectNodesScreenSpace(cam, fovY, aspect, screenH, thr);
//                auto en = std::chrono::high_resolution_clock::now();
//                double ms = std::chrono::duration<double, std::milli>(en - st).count();
//
//                // histogram
//                std::vector<size_t> hist(svo.levels, 0);
//                for(uint32_t id : sel) hist[svo.nodes[id].level]++;
//
//                // print summary
//                size_t totalSel = sel.size();
//                fmt::print("Cam {} thr {:4.1f} -> sel {} nodes in {:.3f} ms | levels:", (int)ci, thr, totalSel, ms);
//                for(int L = int(svo.levels) - 1; L >= 0; --L) {
//                    if(hist[L] == 0) continue;
//                    fmt::print(" [L{}={}]", (int)L, (int)hist[L]);
//                }
//                fmt::println("");
//            }
//        }
//        };
//
//    runProbe(svoNo, "NoCollapse");
//    runProbe(svoCollapsed, "Collapsed+Compacted");
//
//    // Correctness test via sampling: pick random filled voxel coords and ensure collapsed SVO can cover them
//    // Sample up to 200 voxels
//    std::mt19937 rng(123456);
//    std::uniform_int_distribution<int> dx(0, size - 1), dy(0, size - 1), dz(0, size - 1);
//
//    int checks = 200;
//    int foundCount = 0;
//    for(int i = 0; i < checks; ++i) {
//        int x = dx(rng), y = dy(rng), z = dz(rng);
//        uint8_t v = grid[x + y * size + z * size * size];
//        if(v == 0) { --i; continue; } // ensure filled
//        glm::vec3 worldCenter = (glm::vec3(float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f) / float(size)) * (wu - wl) + wl;
//
//        // run selection with a very small pixelThreshold so selected nodes include leaf-level nodes around the point
//        auto sel = svoCollapsed.selectNodesScreenSpace(worldCenter + glm::vec3(0.0f, 0.0f, 0.0f), fovY, aspect, screenH, 2.0f);
//
//        // find a selected node that contains this center
//        bool ok = false;
//        for(uint32_t id : sel) {
//            const auto& n = svoCollapsed.nodes[id];
//            glm::vec3 mn, mx;
//            svoCollapsed.computeWorldAABB(n, mn, mx);
//            if(worldCenter.x >= mn.x && worldCenter.x <= mx.x &&
//                worldCenter.y >= mn.y && worldCenter.y <= mx.y &&
//                worldCenter.z >= mn.z && worldCenter.z <= mx.z) {
//                // must have non-zero color
//                if(n.color != 0 || n.brickIndex >= 0) { ok = true; break; }
//            }
//        }
//        if(ok) ++foundCount;
//    }
//
//    fmt::println("Coverage sampling: {}/{} sampled filled voxels had a containing selected node in collapsed SVO", foundCount, checks);
//    assert(foundCount == checks && "All sampled filled voxels must be contained by a selected node in collapsed SVO");
//
//    fmt::println("\nComprehensive SVO Test: PASSED");
//}
//
//void SVOUnitTests::voxelWorldCompactionTest()
//{
//    // ===== Create a large sparse world =====
//    constexpr int SIZE = 1024; // Grid size
//    std::vector<uint8_t> grid(SIZE * SIZE * SIZE, 0);
//
//    // Ground parameters
//    const int minGroundHeight = 50;
//    const int maxGroundHeight = 300;
//
//    // Cloud parameters
//    const int cloudStartY = 700;
//    const int cloudThickness = 32;
//    const uint8_t cloudColor = 2;
//
//    // Random terrain seed
//    std::mt19937 rng(12345);
//    std::uniform_int_distribution<int> heightDist(minGroundHeight, maxGroundHeight);
//
//    // Generate terrain (mountains)
//    for(int x = 0; x < SIZE; ++x) {
//        for(int z = 0; z < SIZE; ++z) {
//            int h = heightDist(rng);
//            for(int y = 0; y <= h; ++y) {
//                grid[x + y * SIZE + z * SIZE * SIZE] = 1; // rock
//            }
//        }
//    }
//
//    // Generate cloud slabs
//    for(int x = 100; x < 924; x += 64) {
//        for(int z = 100; z < 924; z += 64) {
//            for(int y = cloudStartY; y < cloudStartY + cloudThickness; ++y) {
//                for(int dx = 0; dx < 32; ++dx) {
//                    for(int dz = 0; dz < 32; ++dz) {
//                        grid[(x + dx) + y * SIZE + (z + dz) * SIZE * SIZE] = cloudColor;
//                    }
//                }
//            }
//        }
//    }
//
//    glm::uvec3 gridSize(SIZE, SIZE, SIZE);
//    glm::vec3 worldLower(0.0f);
//    glm::vec3 worldUpper(static_cast<float>(SIZE));
//
//    // ===== Build NON-COLLAPSED SVO =====
//    fmt::print("Building NON-COLLAPSED SVO...\n");
//    SVO svoNoCollapse(grid, gridSize, worldLower, worldUpper, false);
//    auto memNoCollapse = svoNoCollapse.estimateMemoryUsageBytes();
//    fmt::print("NON-COLLAPSED: nodes={}, bricks={}, mem={} bytes\n",
//        svoNoCollapse.nodes.size(),
//        svoNoCollapse.bricks.size(),
//        memNoCollapse);
//
//    // ===== Build COLLAPSED SVO =====
//    fmt::print("Building COLLAPSED SVO...\n");
//    SVO svoCollapse(grid, gridSize, worldLower, worldUpper, true);
//    auto memCollapsed = svoCollapse.estimateMemoryUsageBytes();
//    fmt::print("COLLAPSED: nodes={}, bricks={}, mem={} bytes\n",
//        svoCollapse.nodes.size(),
//        svoCollapse.bricks.size(),
//        memCollapsed);
//
//    // ===== Compact COLLAPSED SVO =====
//    svoCollapse.compact(); // reclaim memory
//    auto memCompact = svoCollapse.estimateMemoryUsageBytes();
//    fmt::print("COLLAPSED + COMPACTED: nodes={}, bricks={}, mem={} bytes\n",
//        svoCollapse.nodes.size(),
//        svoCollapse.bricks.size(),
//        memCompact);
//
//    fmt::print("Memory savings ratio: {:.2f}x\n", float(memNoCollapse) / float(memCompact));
//
//    // ===== LOD Selection Benchmark =====
//    std::vector<glm::vec3> cameras = {
//        {512.0f, 1024.0f, 512.0f}, // overhead
//        {0.0f, 150.0f, 0.0f},      // corner
//        {1024.0f, 150.0f, 1024.0f}, // opposite corner
//        {512.0f, 150.0f, 0.0f}     // mid-edge
//    };
//
//    float pixelThresholds[] = { 1.0f, 4.0f, 16.0f, 32.0f };
//
//    for(float pixThr : pixelThresholds) {
//        fmt::print("\n--- LOD Selection Benchmark (pixelThreshold = {}) ---\n", pixThr);
//
//        for(size_t c = 0; c < cameras.size(); ++c) {
//            auto camPos = cameras[c];
//
//            // Non-collapsed
//            auto selectedNo = svoNoCollapse.selectNodesScreenSpace(camPos, glm::radians(60.0f), 16.0f / 9.0f, 1080, pixThr);
//            fmt::print("Camera {} (NoCollapse) -> selected {} nodes\n", c, selectedNo.size());
//
//            // Collapsed
//            auto selectedCol = svoCollapse.selectNodesScreenSpace(camPos, glm::radians(60.0f), 16.0f / 9.0f, 1080, pixThr);
//            fmt::print("Camera {} (Collapsed) -> selected {} nodes\n", c, selectedCol.size());
//        }
//    }
//}

void SVOUnitTests::benchmark()
{
    fmt::print("Running SVO tests...\n");

    try 
    {
        testBrickedCorrectness();
        testBrickedCorrectnessNonPowerOfTwo();
        testBrickedEfficiency();
        benchmarkLODSimulation();
        benchmarkFineLODSelection();
        //benchmarkLargeScaleEfficiency();
        testLargeWorldScreenSpaceLOD();
        // Tests for possible node compaction. Did not work well though
        //testVoxelCoverage();
        //testScreenSpaceLOD();
        //testScreenSpaceLODHistogram();
        //testCompactionMemorySavings();
        //runComprehensiveSVOTest();
        //voxelWorldCompactionTest();

        fmt::print("\nAll tests passed!\n");
    }
    catch(const std::exception& e) 
    {
        fmt::print(stderr, "Test failed with exception: {}\n", e.what());
        throw;
    }
}