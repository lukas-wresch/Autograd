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

		Storage(const std::vector<float>& Data) : m_Size(Data.size()), m_Data(new float[Data.size()])
		{
			std::copy(Data.begin(), Data.end(), m_Data);
		}

		float* SetData() { return m_Data; }
		const float* GetData() const { return m_Data; }

		size_t GetSize() const { return m_Size; }

	private:
		size_t m_Size;
		float* m_Data;
	};



	Tensor(std::vector<size_t> Shape) : shape_(Shape)
	{
		size_t sum = 0;
		for (auto s : Shape)
			sum += s;
		storage_ = new Storage(sum);

		strides_ = ComputeStrides(Shape);
	}

	Tensor(const std::vector<float>& Data, const std::vector<size_t>& Shape): storage_(new Storage(Data)), shape_(Shape)
	{
		size_t expected = 1;
		for (auto s : shape_) expected *= s;

		if (expected != Data.size())
			throw std::runtime_error("Shape does not match data size");

		strides_ = ComputeStrides(shape_);
	}

	Tensor View(const std::vector<size_t>& new_shape) const
	{
		return Tensor(storage_, new_shape, strides_);
	}

private:
	Tensor(Storage* Storage, const std::vector<size_t>& Shape) : storage_(Storage), shape_(Shape)
	{
		strides_ = ComputeStrides(Shape);
	}

	Tensor(Storage* Storage, const std::vector<size_t>& Shape, const std::vector<size_t>& Strides) : storage_(Storage), shape_(Shape), stides_(Strides)
	{}


	std::vector<size_t> ComputeStrides(const std::vector<size_t>& shape)
	{
		std::vector<size_t> strides(shape.size());
		size_t cur_stride = 1;
		for (int i = shape.size() - 1; i >= 0; i--)
		{
			strides[i] = cur_stride;
			cur_stride *= shape[i];
		}
		return strides;
	}


	Storage* storage_ = nullptr;
	std::vector<size_t> shape_;
	std::vector<size_t> strides_;
	size_t offset_ = 0;
};