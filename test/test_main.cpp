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

TEST_CASE("numRowPart = 1, identical to v1", "[matvec v2]") 
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

TEST_CASE("Different numRowParts and dimensions", "[matvec v2]") 
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
            const int numRowParts = pow(2, m);
            if(numRowParts <= dim)
            {
                args[2] = std::to_string(dim);
                args[3] = std::to_string(numRowParts);
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
    args[2] = "3"; // Dimension
    args[3] = "1"; // rowBlockSize
    args[4] = "1"; // columnBlockSize
    for(auto n : powers)
    {
        for(auto m : powers)
        {
            for(auto p : powers)
            {
                const int dim = pow(2, n);
                const int rowBlockSize = pow(2, m);
                const int columnBlockSize = pow(2, p);
                if (rowBlockSize <= dim && columnBlockSize <= dim && rowBlockSize*columnBlockSize < 256)
                {
                    std::cout << dim << " " << rowBlockSize << " " << columnBlockSize << std::endl;
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