#pragma once
#include <initializer_list>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <functional>
#include <iostream>
#include <iomanip>
#include "matrix.h"



class Tensor4D
{
public:
	class Storage
	{
	public:
		Storage(size_t Size) : m_Size(Size), m_Data(new float[Size])
		{
			std::fill(m_Data, m_Data + m_Size, 0.0f);
		}

		Storage(size_t Size, const float* Data) : m_Size(Size), m_Data(new float[Size])
		{
			std::copy(Data, Data + Size, m_Data);
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



	struct Shape
	{
		Shape() = default;

		Shape(std::vector<size_t> Shape)
		{
			for (size_t i = 0; i < 4; i++)
				size[i] = (i < Shape.size()) ? Shape[i] : 1;

			ComputeStrides();
		}

		void operator=(const Shape& S)
		{
			for (size_t i = 0; i < 4; i++)
			{
				size[i]   = S.size[i];
				stride[i] = S.stride[i];
			}
		}

		size_t ComputeSize() const
		{
			size_t s = 1;
			for (size_t i = 0; i < 4; i++)
				s *= size[i];
			return s;
		}

		void ComputeStrides()
		{
			size_t cur_stride = 1;
			for (int i = 3; i >= 0; i--)
			{
				stride[i] = cur_stride;
				cur_stride *= size[i];
			}
		}

		size_t Index(size_t b, size_t d, size_t i, size_t j) const
		{
			return b * stride[0] + d * stride[1] + i * stride[2] + j * stride[3];
		}

		size_t& operator[](size_t idx)
		{
			if (idx >= 4)
				throw std::out_of_range("Shape index out of bounds");

			return size[idx];
		}

		const size_t& operator[](size_t idx) const
		{
			if (idx >= 4)
				throw std::out_of_range("Shape index out of bounds");

			return size[idx];
		}

		size_t size[4];
		size_t stride[4];
	};



	Tensor4D(const Tensor4D& T) : m_Storage(T.m_Storage), m_Shape(T.m_Shape), m_Offset(T.m_Offset)
	{}

	~Tensor4D()
	{}

	void operator=(const Tensor4D& T)
	{
		m_Storage = T.m_Storage;
		m_Shape   = T.m_Shape;
		m_Offset  = T.m_Offset;
	}

	Tensor4D(Tensor4D&&) noexcept = default;
	Tensor4D& operator=(Tensor4D&&) noexcept = default;

	Tensor4D(std::vector<size_t> Shape) : m_Shape(Shape)
	{
		m_Storage = std::make_shared<Storage>(m_Shape.ComputeSize());
	}

	Tensor4D(const std::vector<size_t>& Shape, const std::vector<float>& Data): m_Storage(new Storage(Data)), m_Shape(Shape)
	{
		if (m_Shape.ComputeSize() != Data.size())
			throw std::runtime_error("Shape does not match data size");
	}

	Tensor4D(const std::vector<size_t>& Shape, const float* Data) : m_Shape(Shape)
	{
		m_Storage = std::make_shared<Storage>(m_Shape.ComputeSize(), Data);
	}

	Tensor4D Reshape(const Shape& newShape) const
	{
		if (newShape.ComputeSize() != m_Shape.ComputeSize())
			throw std::runtime_error("reshape size mismatch");

		Tensor4D out;

		out.m_Storage = m_Storage;
		out.m_Shape = newShape;

		return out;
	}

	/*Tensor4D View(const std::vector<size_t>& new_shape) const
	{
		return Tensor4D(m_Storage, new_shape);
	}*/

	/*Tensor4D ViewRow(size_t batch, size_t depth, size_t row) const
	{
		if (row >= GetRow())
			throw std::out_of_range("Row index out of range");

		// neue Shape: 1 x cols (eine Zeile)
		std::vector<size_t> new_shape = { 1, 1, 1, m_Shape[1] };

		// gleiche Strides behalten, aber effektiv nur Spalten relevant
		std::vector<size_t> new_strides = { 0, m_Shape.stride[1] };

		// Offset springt auf die gewünschte Zeile
		size_t new_offset = m_Offset + row * m_Shape.stride[0];

		return Tensor4D(m_Storage, new_shape, new_strides, new_offset);
	}

	Tensor4D ViewColumn(size_t col) const
	{
		return Tensor(m_Storage, { m_Shape[0], 1 }, { m_Strides[0], 0 }, m_Offset + col);
	}*/

	size_t Dim() const { return 4; }
	const size_t* Strides() const { return m_Shape.stride; }
	size_t Offset() const { return m_Offset; }

	size_t GetSize() const
	{
		return m_Shape.ComputeSize();
	}

	void SetZero()
	{
		std::fill(Data(), Data() + GetSize(), 0.0f);
	}

	void SetOne()
	{
		std::fill(Data(), Data() + GetSize(), 1.0f);
	}

	float* Data()
	{
		return m_Storage->SetData() + m_Offset;
	}

	const float* Data() const
	{
		return m_Storage->GetData() + m_Offset;
	}

	/*inline size_t Index(const std::vector<size_t>& idx) const
	{
		size_t offset = m_Offset;

		for (size_t i = 0; i < 4; i++)
			offset += idx[i] * m_Strides[i];

		return offset;
	}

	std::vector<size_t> BroadcastStrides(const std::vector<size_t>& outShape) const
	{
		std::vector<size_t> result(outShape.size(), 0);

		size_t ndimDiff = outShape.size() - Dim();

		for (size_t i = 0; i < Dim(); i++)
		{
			if (m_Shape[i] == 1)
				result[i + ndimDiff] = 0;
			else
				result[i + ndimDiff] = m_Strides[i];
		}

		return result;
	}*/

	float At(size_t b, size_t d, size_t r, size_t c) const
	{
		if (b >= m_Shape[0] || d >= m_Shape[1] || r >= m_Shape[2] || c >= m_Shape[3])
			throw std::out_of_range("Tensor4D::At(b,d,r,c) index out of bounds");

		size_t idx = m_Shape.Index(b, d, r, c);
		return m_Storage->GetData()[m_Offset + idx];
	}

	float& At(size_t b, size_t d, size_t r, size_t c)
	{
		if (b >= m_Shape[0] || d >= m_Shape[1] || r >= m_Shape[2] || c >= m_Shape[3])
			throw std::out_of_range("Tensor4D::At(b,d,r,c) index out of bounds");

		size_t idx = m_Shape.Index(b, d, r, c);
		return m_Storage->SetData()[m_Offset + idx];
	}

	/*void Transpose()
	{
		if (Dim() != 2)
			throw std::runtime_error("Transpose only valid for 2D tensors");

		std::swap(m_Shape[0], m_Shape[1]);
		std::swap(m_Strides[0], m_Strides[1]);
	}*/

	size_t GetBatches() const
	{
		return m_Shape[0];
	}

	size_t GetDepth() const
	{
		return m_Shape[1];
	}

	size_t GetRows() const
	{
		return m_Shape[2];
	}

	size_t GetColumns() const
	{
		return m_Shape[3];
	}

	std::string Shape2String() const
	{
		return std::to_string(m_Shape[0]) + "x" + std::to_string(m_Shape[1]) + "x" + std::to_string(m_Shape[2]) + "x" + std::to_string(m_Shape[3]);
	}

	void Print() const
	{
		std::cout << "Tensor (" << Shape2String() << ")\n";
		std::cout << "Offset: " << m_Offset << "\n";
		std::cout << "Strides: ";

		for (size_t i = 0; i < 4; i++)
		{
			std::cout << m_Shape.stride[i];
			if (i + 1 < 4) std::cout << ", ";
		}
		std::cout << "\nValues:\n";

		// ND fallback (flattened view)
		std::cout << "[flattened]\n[ ";
		for (size_t i = 0; i < GetSize(); i++)
		{
			std::cout << std::fixed << std::setprecision(4) << Data()[i] << " ";
		}
		std::cout << "]\n";
	}

	void SetRow(size_t batch, size_t depth, size_t row, std::vector<float> vec)
	{
		size_t cols = GetColumns();

		if (vec.size() != cols)
			throw std::runtime_error("Row size mismatch");

		for (size_t c = 0; c < cols; c++)
			At(batch, depth, row, c) = vec[c];
	}

	void SetColumn(size_t batch, size_t depth, size_t col, const std::vector<float>& vec)
	{
		size_t rows = GetRows();

		if (vec.size() != rows)
			throw std::runtime_error("Column size mismatch");

		for (size_t r = 0; r < rows; r++)
			At(batch, depth, r, col) = vec[r];
	}


	Tensor Sum() const
	{
		float sum = 0.0f;
		for (size_t i = 0; i < GetSize(); i++)
			sum += Data()[i];
		return Tensor({ 1 }, { sum });
	}

	float Max() const
	{
		float max = Data()[0];
		for (size_t i = 1; i < GetSize(); i++)
		{
			if (Data()[i] > max)
				max = Data()[i];
		}
		return max;
	}

	Tensor4D Tanh() const
	{
		Tensor4D out(*this);
		for (size_t i = 0; i < GetSize(); i++)
			out.Data()[i] = std::tanh(Data()[i]);
		return out;
	}

	Tensor4D ReLU() const
	{
		Tensor4D out(*this);
		for (size_t i = 0; i < GetSize(); i++)
			out.Data()[i] = std::fmax(0.0f, Data()[i]);
		return out;
	}

	//Derivate of ReLU
	Tensor4D Heaviside() const
	{
		Tensor4D out(*this);
		for (size_t i = 0; i < GetSize(); i++)
			out.Data()[i] = (Data()[i] > 0.0f ? 1.0f : 0.0f);
		return out;
	}

	inline Tensor4D ElementwiseMul(const Tensor& rhs) const;

	/*Tensor4D Softmax() const
	{
		if (m_Shape.size() > 2)
			throw std::runtime_error("Only vector/matrix type supported");

		float max_value = Max();

		Tensor4D out(Shape());

		float sum = 0.0f;
		for (size_t i = 0; i < GetSize(); i++)
		{
			out.Data()[i] = std::exp(Data()[i] - max_value);
			sum += out.Data()[i];
		}

		for (size_t i = 0; i < GetSize(); i++)
			out.Data()[i] /= sum;

		return out;
	}

	Tensor4D CrossEntropy(const Tensor4D& Target) const
	{
		const float epsilon = 0.0001f;
		float loss = 0.0f;

		//Target only contains the label (Sparse)
		if (Target.Size() == GetColumns())
		{
			for (size_t i = 0; i < GetColumns(); i++)
			{
				int label = (int)Target.At({ i });

				if (label < 0 || label >= (int)GetRows())
					throw std::runtime_error("Tensor CrossEntropy invalid label index");

				float p = At({ (size_t)label, i });
				loss -= std::log(p + epsilon);
			}

			return Tensor4D({ 1 }, { loss / GetColumns() });
		}

		//Target is on-hot vector (Dense)
		if (GetRows() != Target.GetRows() || GetColumns() != Target.GetColumns())
			throw std::runtime_error("Matrix CrossEntropy size mismatch");

		for (size_t i = 0; i < GetRows(); i++)
			for (size_t j = 0; j < GetColumns(); j++)
				loss -= Target.At({ i, j }) * std::log(At({ i, j }) + epsilon);

		return Tensor4D({ 1 }, { loss / GetColumns() });
	}*/


private:
	Tensor4D() {}
	Tensor4D(Storage* Storage, const std::vector<size_t>& Shape) : m_Storage(Storage), m_Shape(Shape)
	{}

	/*Tensor4D(Storage* Storage, const std::vector<size_t>& Shape, const std::vector<size_t>& Strides, size_t Offset = 0)
		: m_Storage(Storage), m_Shape(Shape), m_Offset(Offset)
	{}*/



	std::shared_ptr<Storage> m_Storage;
	Shape m_Shape;
	size_t m_Offset = 0;
};



/*inline void ForEachIndex(const Tensor4D& t, std::function<void(const std::vector<size_t>&)> fn)
{
	std::vector<size_t> idx(t.m_Shape().size(), 0);

	size_t total = t.GetSize();

	for (size_t i = 0; i < total; i++)
	{
		size_t tmp = i;

		for (int d = (int)t.m_Shape().size() - 1; d >= 0; d--)
		{
			idx[d] = tmp % t.m_Shape()[d];
			tmp /= t.m_Shape()[d];
		}

		fn(idx);
	}
}*/



inline std::vector<size_t> Tensor4D_BroadcastShape(const std::vector<size_t>& a, const std::vector<size_t>& b)
{
	size_t maxDim = std::max(a.size(), b.size());

	std::vector<size_t> result(maxDim);

	for (size_t i = 0; i < maxDim; i++)
	{
		size_t ad = (i < a.size()) ? a[a.size() - 1 - i] : 1;
		size_t bd = (i < b.size()) ? b[b.size() - 1 - i] : 1;

		result[maxDim - 1 - i] = std::max(ad, bd);
	}

	return result;
}



inline bool Tensor4D_IsBroadcastCompatible(const std::vector<size_t>& ShapeA, const std::vector<size_t>& ShapeB)
{
	size_t aDim = ShapeA.size();
	size_t bDim = ShapeB.size();
	size_t maxDim = std::max(aDim, bDim);

	for (size_t i = 0; i < maxDim; i++)
	{
		size_t ad = (i < aDim) ? ShapeA[aDim - 1 - i] : 1;
		size_t bd = (i < bDim) ? ShapeB[bDim - 1 - i] : 1;

		if (ad != bd && ad != 1 && bd != 1)
			return false;
	}

	return true;
}