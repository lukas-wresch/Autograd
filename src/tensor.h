#pragma once
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include "matrix.h"



class Tensor
{
public:
	class Storage
	{
	public:
		Storage(size_t Size) : m_Size(Size), m_Data(new float[Size])
		{
			std::fill(m_Data, m_Data + m_Size, 0.0f);
		}

		Storage(const std::vector<float>& Data) : m_Size(Data.size()), m_Data(new float[Data.size()])
		{
			std::copy(Data.begin(), Data.end(), m_Data);
		}

		~Storage()
		{
			delete[] m_Data;
			m_Data = nullptr;
			m_Size = 0;
		}

		float* SetData() { return m_Data; }
		const float* GetData() const { return m_Data; }

		size_t GetSize() const { return m_Size; }

	private:
		size_t m_Size;
		float* m_Data;
	};



	Tensor(Tensor&& T) : storage_(T.storage_), shape_(T.shape_), strides_(T.strides_), offset_(T.offset_)
	{
		T.storage_ = nullptr;
		T.shape_   = {};
		T.strides_ = {};
		T.offset_  = 0;
	}

	Tensor(std::vector<size_t> Shape) : shape_(Shape)
	{
		size_t size = 1;
		for (auto s : Shape)
			size *= s;
		storage_ = new Storage(size);

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

	/*Tensor(const Matrix& Mat)
	{
		shape_ = std::vector<size_t>{ Mat.GetRows(), Mat.GetColumns() };

		size_t size = Mat.GetRows() * Mat.GetColumns();

		storage_ = new Storage(size);

		std::copy(
			Mat.GetValue(),
			Mat.GetValue() + size,
			storage_->SetData()
		);

		strides_ = ComputeStrides(shape_);
	}*/

	Tensor View(const std::vector<size_t>& new_shape) const
	{
		return Tensor(storage_, new_shape, strides_);
	}

	const std::vector<size_t>& Shape() const { return shape_; }
	const std::vector<size_t>& Strides() const { return strides_; }
	size_t Offset() const { return offset_; }

	size_t Size() const
	{
		size_t s = 1;

		for (size_t v : shape_)
			s *= v;

		return s;
	}

	float* Data()
	{
		if (storage_)
			return storage_->SetData() + offset_;
		return nullptr;
	}

	const float* Data() const
	{
		if (storage_)
			return storage_->GetData() + offset_;
		return nullptr;
	}

	inline size_t Offset(const std::vector<size_t>& idx) const
	{
		size_t offset = offset_;

		for (size_t i = 0; i < shape_.size(); i++)
			offset += idx[i] * strides_[i];

		return offset;
	}

	std::vector<size_t> BroadcastStrides(const std::vector<size_t>& outShape) const
	{
		std::vector<size_t> result(outShape.size(), 0);

		size_t ndimDiff = outShape.size() - shape_.size();

		for (size_t i = 0; i < shape_.size(); i++)
		{
			if (shape_[i] == 1)
				result[i + ndimDiff] = 0;
			else
				result[i + ndimDiff] = strides_[i];
		}

		return result;
	}

private:
	Tensor(Storage* Storage, const std::vector<size_t>& Shape) : storage_(Storage), shape_(Shape)
	{
		strides_ = ComputeStrides(Shape);
	}

	Tensor(Storage* Storage, const std::vector<size_t>& Shape, const std::vector<size_t>& Strides) : storage_(Storage), shape_(Shape), strides_(Strides)
	{}


	std::vector<size_t> ComputeStrides(const std::vector<size_t>& shape)
	{
		std::vector<size_t> strides(shape.size());
		size_t cur_stride = 1;
		for (int i = (int)shape.size() - 1; i >= 0; i--)
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