#pragma once

#include <random>
#include <algorithm>

class Matrix
{
    public:
        Matrix(size_t size)
        {
            m_elements.resize(size*size);

            auto prng = [engine = std::default_random_engine{},
                        distribution = std::uniform_real_distribution<cl_float>{ -1.0, 1.0 }]() mutable { return distribution(engine); };

            std::generate_n(m_elements.begin(), m_elements.size(), prng);
            //initializeAsIdentity(size);
        }

        void actsOnVector(const std::vector<cl_float>& inVector, std::vector<cl_float>& outVector)
        {
            size_t size = sqrt(m_elements.size());
            for(size_t row = 0; row < size; row++)
            {
                cl_float result = 0.0f;
                for(size_t column = 0; column < size; column++)
                    result += m_elements[row*size + column] * inVector[column];
                outVector[row] = result;
            }
        }

        std::vector<cl_float>& getElements() { return m_elements; }

    private:
        std::vector<cl_float> m_elements;

        void initializeAsIdentity(const size_t size)
        {
            for(int i = 0; i < size; i++)
            {
                for(int j = 0; j < size; j++)
                    m_elements[i*size+j] = i == j ? 1.0f : 0.0f;
            }
        }

};