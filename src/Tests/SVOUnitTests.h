#pragma once

#include <Data/SVO.h>
#include <fmt/core.h>
#include <cassert>
#include <chrono>
#include <vector>
#include <random>

class SVOUnitTests
{
public:
    void benchmark();

private:
    void testBasicCorrectness();
    void testEmptyGrid();
    void testSingleVoxel();
    void test2x2x2Grid();
    void testFullGrid();
    void testSparseGrid();
    void testMemoryUsage();
    void testLODSelection();
    void testLargeSparseGrid();
    void testComplexPattern();
    void testCorrectness();
    void testStructureCorrectness();
    void testLargeScaleTerrainLOD();

    void printTestResult(const std::string& testName, bool passed);
};