#pragma once
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <algorithm>
#include <vector>



class Tensor
{
public:
	class Storage
	{
	public:
		Storage(size_t Size) : m_Size(Size), m_Data(new float[Size])
		{}

		float* data() { return m_Data; }
		const float* data() const { return m_Data; }

		size_t size() const { return m_Size; }

	private:
		size_t m_Size;
		float* m_Data;
	};


private:
	Storage* storage_ = nullptr;
	std::vector<size_t> shape_;
	std::vector<size_t> strides_;
	size_t offset_;
};