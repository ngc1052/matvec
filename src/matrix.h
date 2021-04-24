#pragma once
#include <stddef.h>

class Matrix
{
    public:
        Matrix(size_t size) : m_size(size)
        {
            m_elements = new float[m_size * m_size];

            // Identity matrix
            for(size_t row = 0; row < m_size; row++)
            {
                for(size_t column = 0; column < m_size; column++)
                {
                    if(row == column)
                        m_elements[row*m_size + column] = 1.0;
                    else
                        m_elements[row*m_size + column] = 0.0;
                }
            }
        }

        float* getElements() const { return m_elements; }
        size_t getSize() const { return m_size; }

        ~Matrix()
        {
            delete[] m_elements;
        }

    private:
        size_t m_size;
        float* m_elements;

};