#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "application.h"
#include <iostream>


TEST_CASE("Match with reference algorithm for sizes: 2**n, n = 2, 3, ..., 10", "[matvec v1]") 
{
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

TEST_CASE("rowBlockSize = 1, identical to v1", "[matvec v2]") 
{
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

TEST_CASE("Different rowBlockSize and dimensions", "[matvec v2]") 
{
    std::vector<int> powers = {2, 3, 4, 5, 6};
    std::vector<std::string> args(4);
    args[0] = "";
    args[1] = "2";
    for(auto n : powers)
    {
        for(auto m : powers)
        {
            const int dim = pow(2, n);
            const int rowBlockSize = pow(2, m);
            if(rowBlockSize <= dim)
            {
                args[2] = std::to_string(dim);
                args[3] = std::to_string(rowBlockSize);
                Application app;
                app.run_v2(args);
                REQUIRE(app.matchOutAndReferenceVectors() == true);
            }
        }
    }
}

TEST_CASE("Different dimensions and block sizes", "[matvec v3]") 
{
    std::vector<int> powers = {2, 3, 4, 5, 6};
    std::vector<std::string> args(5);
    args[0] = "";
    args[1] = "3"; // Method 3
    for(auto a : powers)
    {
        for(auto b : powers)
        {
            for(auto c : powers)
            {
                const int dim = pow(2, a);
                const int rowBlockSize = pow(2, b);
                const int columnBlockSize = pow(2, c);
                if (rowBlockSize <= dim && columnBlockSize <= dim && rowBlockSize*columnBlockSize < 256)
                {
                    args[2] = std::to_string(dim);
                    args[3] = std::to_string(rowBlockSize);
                    args[4] = std::to_string(columnBlockSize);
                    Application app;
                    app.run_v3(args);
                    REQUIRE(app.matchOutAndReferenceVectors() == true);
                }
            }
        }
    }
}