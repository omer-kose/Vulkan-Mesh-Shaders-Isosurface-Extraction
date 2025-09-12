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
    void testBrickedEfficiency();
    void benchmarkLODSimulation();
    void benchmarkLargeScaleEfficiency();
    void benchmarkFineLODSelection();

    void printTestResult(const std::string& testName, bool passed);
};