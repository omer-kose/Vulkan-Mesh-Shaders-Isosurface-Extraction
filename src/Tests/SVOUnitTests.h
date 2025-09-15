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
    void testBrickedCorrectness();
    void testBrickedCorrectnessNonPowerOfTwo();
    void testBrickedEfficiency();
    void benchmarkLODSimulation();
    void benchmarkLargeScaleEfficiency();
    void benchmarkFineLODSelection();
    void testLargeWorldScreenSpaceLOD();
    //void testVoxelCoverage();
    //void testScreenSpaceLOD();
    //void testScreenSpaceLODHistogram();
    //void testCompactionMemorySavings();
    //void runComprehensiveSVOTest();
    //void voxelWorldCompactionTest();

    void printTestResult(const std::string& testName, bool passed);
};