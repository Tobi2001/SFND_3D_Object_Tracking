#ifndef RINGBUFFER_H
#define RINGBUFFER_H

#include <array>


template<class DataT, int BufferSizeT>
class Ringbuffer
{
    static_assert(BufferSizeT > 0, "Ringbuffer size <= 0");

public:
    Ringbuffer()
      : m_size(0),
        m_indexBack(0)
    {}

    size_t size() const
    {
        return m_size;
    }

    void push_back(const DataT& element)
    {
        if (m_size > 0u)
        {
            m_indexBack = (m_indexBack + 1) % BufferSizeT;
        }
        m_data[m_indexBack] = element;
        if (m_size < m_data.size())
        {
            ++m_size;
        }
    }

    DataT& at(const int index)
    {
        if (index < m_size)
        {
            int currIndex = m_indexBack - index;
            if (currIndex < 0)
            {
                currIndex += BufferSizeT;
            }
            return m_data[currIndex];
        }
        throw std::runtime_error("Index is out of bounds");
    }

private:
    // Fixed array is sufficient for the project.
    // Dynamically allocated memory would be better though in general applications
    std::array<DataT, BufferSizeT> m_data;
    size_t m_size;
    size_t m_indexBack;
};

#endif // RINGBUFFER_H
