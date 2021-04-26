#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "application.h"


TEST_CASE("Match with reference algorithm for sizes: 2**n, n = 2, 3, ..., 10", "[matvec v1]") {
    std::vector<int> powers = {2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<std::string> args(3);
    args[0] = "";
    args[1] = "1";
    for(auto n : powers)
    {
        const int dim = pow(2, n);
        args[2] = std::to_string(dim);
        Application app;
        app.run_v1(args);
        REQUIRE(app.matchOutAndReferenceVectors() == true);
    }
}

TEST_CASE("numRowPart = 1, identical to v1", "[matvec v2]") {
    std::vector<int> powers = {2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<std::string> args(4);
    args[0] = "";
    args[1] = "2";
    args[3] = "1";
    for(auto n : powers)
    {
        const int dim = pow(2, n);
        args[2] = std::to_string(dim);
        Application app;
        app.run_v2(args);
        REQUIRE(app.matchOutAndReferenceVectors() == true);
    }
}

TEST_CASE("Different numRowParts and dimensions", "[matvec v2]") {
    std::vector<int> powers = {2, 3, 4, 5, 6};
    std::vector<std::string> args(4);
    args[0] = "";
    args[1] = "2";
    args[3] = "1";
    for(auto n : powers)
    {
        for(auto m : powers)
        {
            const int dim = pow(2, n);
            const int numRowParts = pow(2, m);
            if(numRowParts <= dim)
            {
                args[2] = std::to_string(dim);
                Application app;
                app.run_v2(args);
                REQUIRE(app.matchOutAndReferenceVectors() == true);
            }
        }
    }
}