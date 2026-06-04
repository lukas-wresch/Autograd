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
			for (size_t i = 0; i < 4 - Shape.size(); i++)
				size[i] = 1;
			for (size_t i = 0; i < Shape.size(); i++)
				size[i + (4 - Shape.size())] = Shape[i];

			ComputeStrides();
		}

		Shape(std::vector<size_t> Shape, std::vector<size_t> Strides)
		{
			for (size_t i = 0; i < 4 - Shape.size(); i++)
				size[i] = 1;
			for (size_t i = 0; i < Shape.size(); i++)
				size[i + (4 - Shape.size())] = Shape[i];

			for (size_t i = 0; i < 4; i++)
				stride[i] = Strides[i];
		}

		Shape& operator=(const Shape& S)
		{
			for (size_t i = 0; i < 4; i++)
			{
				size[i]   = S.size[i];
				stride[i] = S.stride[i];
			}
			return *this;
		}

		bool operator==(const Shape& S) const
		{
			for (size_t i = 0; i < 4; i++)
			{
				if (size[i] != S.size[i])
					return false;
			}
			return true;
		}

		bool operator!=(const Shape& S) const
		{
			for (size_t i = 0; i < 4; i++)
			{
				if (size[i] != S.size[i])
					return true;
			}
			return false;
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

		size_t Dim() const
		{
			size_t dim = 0;
			for (size_t i = 0; i < 4; i++)
			{
				if (size[i] > 1)
					dim++;
			}
			return dim;
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

		bool IsContiguous() const
		{
			size_t expected_stride = 1;

			for (int i = 3; i >= 0; --i)
			{
				if (stride[i] != expected_stride)
					return false;

				expected_stride *= size[i];
			}

			return true;
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

	Tensor4D(const Shape& Shape) : m_Shape(Shape)
	{
		m_Storage = std::make_shared<Storage>(m_Shape.ComputeSize());
	}

	Tensor4D(const std::vector<size_t>& Shape, const std::vector<float>& Data): m_Storage(std::make_shared<Storage>(Data)), m_Shape(Shape)
	{
		if (m_Shape.ComputeSize() != Data.size())
			throw std::runtime_error("Shape does not match data size");
	}

	Tensor4D(const std::vector<size_t>& Shape, const float* Data) : m_Shape(Shape)
	{
		m_Storage = std::make_shared<Storage>(m_Shape.ComputeSize(), Data);
	}

	bool IsContiguous() const { return m_Shape.IsContiguous(); }

	float operator[](size_t Index) const
	{
		if (Index >= GetSize())
			throw std::out_of_range("Index out of bounds");

		if (!IsContiguous())
			throw std::out_of_range("Not supported for non-contiguous tensors");

		return m_Storage->GetData()[m_Offset + Index];
	}

	float& operator[](size_t Index)
	{
		if (Index >= GetSize())
			throw std::out_of_range("Index out of bounds");

		if (!IsContiguous())
			throw std::out_of_range("Not supported for non-contiguous tensors");

		return m_Storage->SetData()[m_Offset + Index];
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

	Tensor4D ViewRow(size_t batch, size_t depth, size_t row) const
	{
		if (row >= GetRows())
			throw std::out_of_range("Row index out of range");

		// neue Shape: 1 x cols (eine Zeile)
		Shape new_shape = Shape({1, 1, 1, GetColumns()}, // New shape
								{ 0, 0, 0, m_Shape.stride[3] }); // New strides

		// Offset springt auf die gewünschte Zeile
		size_t new_offset = m_Offset + batch * m_Shape.stride[0] + depth * m_Shape.stride[1] + row * m_Shape.stride[2];

		return Tensor4D(m_Storage, new_shape, new_offset);
	}

	Tensor4D ViewColumn(size_t batch, size_t depth, size_t col) const
	{
		Shape new_shape = Shape({ 1, 1, GetRows(), 1}, // New shape
			{ 0, 0, m_Shape.stride[2], 0 }); // New strides

		size_t new_offset = m_Offset + batch * m_Shape.stride[0] + depth * m_Shape.stride[1] + col * m_Shape.stride[3];

		return Tensor4D(m_Storage, new_shape, new_offset);
	}

	Tensor4D ViewColumn(size_t Column) const
	{
		return ViewColumn(0, 0, Column);
	}

	size_t Dim() const { return m_Shape.Dim(); }
	const size_t* Strides() const { return m_Shape.stride; }
	size_t Offset() const { return m_Offset; }

	size_t GetSize() const
	{
		return m_Shape.ComputeSize();
	}

	void SetZero()
	{
		if (!IsContiguous())
			throw std::out_of_range("Not supported for non-contiguous tensors");
		std::fill(Data(), Data() + GetSize(), 0.0f);
	}

	void SetOne()
	{
		if (!IsContiguous())
			throw std::out_of_range("Not supported for non-contiguous tensors");
		std::fill(Data(), Data() + GetSize(), 1.0f);
	}

	void SetIdentity()
	{
		SetZero();

		for (size_t b = 0; b < GetBatches(); b++)
			for (size_t d = 0; d < GetDepth(); d++)
			{
				size_t minDim = std::min(GetRows(), GetColumns());

				for (size_t i = 0; i < minDim; ++i)
					At(b, d, i, i) = 1.0f;
			}
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

	float At(size_t r, size_t c) const
	{
		return At(0, 0, r, c);
	}

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

	float At(const std::vector<size_t>& idx) const
	{
		if (idx[0] >= m_Shape[0] || idx[1] >= m_Shape[1] || idx[2] >= m_Shape[2] || idx[3] >= m_Shape[3])
			throw std::out_of_range("Tensor4D::At(b,d,r,c) index out of bounds");

		size_t index = m_Shape.Index(idx[0], idx[1], idx[2], idx[3]);
		return m_Storage->GetData()[m_Offset + index];
	}

	float& At(const std::vector<size_t>& idx)
	{
		if (idx[0] >= m_Shape[0] || idx[1] >= m_Shape[1] || idx[2] >= m_Shape[2] || idx[3] >= m_Shape[3])
			throw std::out_of_range("Tensor4D::At(b,d,r,c) index out of bounds");

		size_t index = m_Shape.Index(idx[0], idx[1], idx[2], idx[3]);
		return m_Storage->SetData()[m_Offset + index];
	}

	Tensor4D Transpose()
	{
		Shape newShape = m_Shape;

		std::swap(newShape.size[2],   newShape.size[3]);   // rows <-> cols
		std::swap(newShape.stride[2], newShape.stride[3]); // strides anpassen

		return Tensor4D(m_Storage, newShape, m_Offset);
	}

	const Shape& GetShape() const { return m_Shape; }

	size_t GetBatches() const { return m_Shape[0]; }
	size_t GetDepth() const { return m_Shape[1]; }
	size_t GetRows() const { return m_Shape[2]; }
	size_t GetColumns() const { return m_Shape[3]; }

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


	Tensor4D Sum() const
	{
		float sum = 0.0f;
		for (size_t i = 0; i < GetSize(); i++)
			sum += Data()[i];
		return Tensor4D({ 1, 1, 1, 1 }, { sum });
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

	size_t ArgMax(size_t Batch, size_t Depth) const
	{
		if (GetColumns() != 1)
			throw std::runtime_error("ArgMax: invalid columns");

		size_t best_index = 0;
		float best_value = -std::numeric_limits<float>::infinity();

		for (size_t r = 0; r < GetRows(); r++)
		{
			float v = At(Batch, Depth, r, 0);

			if (v > best_value)
			{
				best_value = v;
				best_index = r;
			}
		}

		return best_index;
	}

	Tensor4D Tanh() const
	{
		if (!IsContiguous())
			throw std::out_of_range("Not supported for non-contiguous tensors");

		Tensor4D out(GetShape());
		for (size_t i = 0; i < GetSize(); i++)
			out.Data()[i] = std::tanh(Data()[i]);
		return out;
	}

	Tensor4D ReLU() const
	{
		if (!IsContiguous())
			throw std::out_of_range("Not supported for non-contiguous tensors");

		Tensor4D out(GetShape());
		for (size_t i = 0; i < GetSize(); i++)
			out.Data()[i] = std::fmax(0.0f, Data()[i]);
		return out;
	}

	//Derivate of ReLU
	Tensor4D Heaviside() const
	{
		if (!IsContiguous())
			throw std::out_of_range("Not supported for non-contiguous tensors");

		Tensor4D out(GetShape());
		for (size_t i = 0; i < GetSize(); i++)
			out.Data()[i] = (Data()[i] > 0.0f ? 1.0f : 0.0f);
		return out;
	}

	//inline Tensor4D ElementwiseMul(const Tensor4D& rhs) const;

	Tensor4D Softmax() const
	{
		if (Dim() > 1)
			throw std::runtime_error("Tensor4D::Softmax() Only vector type supported");

		float max_value = Max();

		Tensor4D out(GetShape());

		float sum = 0.0f;
		for (size_t r = 0; r < GetRows(); r++)
		{
			out[r] = std::exp(At(0, 0, r, 0) - max_value);
			sum += out[r];
		}

		for (size_t i = 0; i < GetRows(); i++)
			out[i] /= sum;

		return out;
	}

	Tensor4D CrossEntropy(const Tensor4D& Target) const
	{
		const float epsilon = 0.0001f;
		float loss = 0.0f;

		if (GetBatches() != Target.GetBatches())
			throw std::runtime_error("Tensor4D CrossEntropy size mismatch");

		// Target only contains the label (Sparse)
		if (Target.GetRows() == 1)
		{
			for (size_t b = 0; b < GetBatches(); b++)
			{
				int label = (int)Target.At(b, 0, 0, 0);

				if (label < 0 || label >= (int)GetRows())
					throw std::runtime_error("Tensor4D CrossEntropy invalid label index");

				float p = At(b, 0, (size_t)label, 0);
				loss -= std::log(p + epsilon);
			}

			return Tensor4D({ 1 }, { loss / GetBatches() });
		}

		// Target is on-hot vector (Dense)
		if (GetRows() != Target.GetRows())
			throw std::runtime_error("Tensor4D CrossEntropy size mismatch");

		for (size_t b = 0; b < GetBatches(); b++)
			for (size_t r = 0; r < GetRows(); r++)
				loss -= Target.At(b, 0, r, 0) * std::log(At(b, 0, r, 0) + epsilon);

		return Tensor4D({ 1 }, { loss / GetBatches() });
	}

	inline void operator+=(const Tensor4D& right);
	inline void operator-=(const Tensor4D& right);
	inline void operator*=(const Tensor4D& right);

	inline void ForEachIndex(std::function<void(size_t)> fn)
	{
		for (size_t b = 0; b < GetBatches(); b++)
			for (size_t d = 0; d < GetDepth(); d++)
				for (size_t r = 0; r < GetRows(); r++)
					for (size_t c = 0; c < GetColumns(); c++)
						fn(m_Shape.Index(b, d, r, c));
		}
	}

private:
	Tensor4D() {}
	Tensor4D(std::shared_ptr<Storage> Storage, const Shape& Shape, size_t Offset = 0) : m_Storage(Storage), m_Shape(Shape), m_Offset(Offset)
	{}

	/*Tensor4D(Storage* Storage, const std::vector<size_t>& Shape, const std::vector<size_t>& Strides, size_t Offset = 0)
		: m_Storage(Storage), m_Shape(Shape), m_Offset(Offset)
	{}*/	


	std::shared_ptr<Storage> m_Storage;
	Shape m_Shape;
	size_t m_Offset = 0;
};



inline std::vector<size_t> Tensor4D_BroadcastShape(const Tensor4D::Shape& a, const Tensor4D::Shape& b)
{
	size_t maxDim = std::max(a.Dim(), b.Dim());

	std::vector<size_t> result(maxDim);

	for (size_t i = 0; i < maxDim; i++)
	{
		size_t ad = (i < a.Dim()) ? a[a.Dim() - 1 - i] : 1;
		size_t bd = (i < b.Dim()) ? b[b.Dim() - 1 - i] : 1;

		result[maxDim - 1 - i] = std::max(ad, bd);
	}

	return result;
}



inline bool Tensor4D_IsBroadcastCompatible(const Tensor4D::Shape& ShapeA, const Tensor4D::Shape& ShapeB)
{
	size_t aDim = ShapeA.Dim();
	size_t bDim = ShapeB.Dim();
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



inline Tensor4D operator+(const Tensor4D& left, const Tensor4D& right)
{
	if (!Tensor4D_IsBroadcastCompatible(left.GetShape(), right.GetShape()))
		throw std::runtime_error("broadcast mismatch");

	auto outShape = Tensor4D_BroadcastShape(left.GetShape(), right.GetShape());

	Tensor4D result(outShape);

	result.ForEachIndex([&](const std::vector<size_t>& idx)
	{
		std::vector<size_t> lidx(left.Dim());
		std::vector<size_t> ridx(right.Dim());

		// map output index -> left/right broadcast index

		int offsetL = (int)outShape.size() - (int)left.Dim();
		int offsetR = (int)outShape.size() - (int)right.Dim();

		for (size_t d = 0; d < left.Dim(); d++)
		{
			if (left.GetShape()[d] == 1)
				lidx[d] = 0;
			else
				lidx[d] = idx[d + offsetL];
		}

		for (size_t d = 0; d < right.Dim(); d++)
		{
			if (right.GetShape()[d] == 1)
				ridx[d] = 0;
			else
				ridx[d] = idx[d + offsetR];
		}

		result.At(idx) = left.At(lidx) + right.At(ridx);
	});

	return result;
}



inline void Tensor4D::operator+=(const Tensor4D& right)
{
	// Scalar case
	if (right.GetSize() == 1)
	{
		ForEachIndex([&](const std::vector<size_t>& idx)
		{
			At(idx) += right[0];
		});

		return;
	}



	if (GetShape() != right.GetShape())
		throw std::runtime_error("incompatible shapes");

	ForEachIndex([&](const std::vector<size_t>& idx)
	{
		std::vector<size_t> lidx(Dim());
		std::vector<size_t> ridx(right.Dim());

		// map output index -> left/right broadcast index

		int offsetL = (int)4 - (int)Dim();
		int offsetR = (int)4 - (int)right.Dim();

		for (size_t d = 0; d < Dim(); d++)
		{
			if (GetShape()[d] == 1)
				lidx[d] = 0;
			else
				lidx[d] = idx[d + offsetL];
		}

		for (size_t d = 0; d < right.Dim(); d++)
		{
			if (right.GetShape()[d] == 1)
				ridx[d] = 0;
			else
				ridx[d] = idx[d + offsetR];
		}

		At(idx) += right.At(ridx);
	});
}



inline Tensor4D operator-(const Tensor4D& left, const Tensor4D& right)
{
	if (!Tensor4D_IsBroadcastCompatible(left.GetShape(), right.GetShape()))
		throw std::runtime_error("broadcast mismatch");

	auto outShape = Tensor4D_BroadcastShape(left.GetShape(), right.GetShape());

	Tensor4D result(outShape);

	result.ForEachIndex([&](const std::vector<size_t>& idx)
	{
		std::vector<size_t> lidx(left.Dim());
		std::vector<size_t> ridx(right.Dim());

		// map output index -> left/right broadcast index

		int offsetL = (int)outShape.size() - (int)left.Dim();
		int offsetR = (int)outShape.size() - (int)right.Dim();

		for (size_t d = 0; d < left.Dim(); d++)
		{
			if (left.GetShape()[d] == 1)
				lidx[d] = 0;
			else
				lidx[d] = idx[d + offsetL];
		}

		for (size_t d = 0; d < right.Dim(); d++)
		{
			if (right.GetShape()[d] == 1)
				ridx[d] = 0;
			else
				ridx[d] = idx[d + offsetR];
		}

		result.At(idx) = left.At(lidx) - right.At(ridx);
	});

	return result;
}



inline void Tensor4D::operator-=(const Tensor4D& right)
{
	// Scalar case
	if (right.GetSize() == 1)
	{
		ForEachIndex([&](const std::vector<size_t>& idx)
		{
			At(idx) -= right[0];
		});

		return;
	}

	if (GetShape() != right.GetShape())
		throw std::runtime_error("incompatible shapes");

	ForEachIndex([&](const std::vector<size_t>& idx)
	{
		std::vector<size_t> lidx(Dim());
		std::vector<size_t> ridx(right.Dim());

		// map output index -> left/right broadcast index

		int offsetL = (int)4 - (int)Dim();
		int offsetR = (int)4 - (int)right.Dim();

		for (size_t d = 0; d < Dim(); d++)
		{
			if (GetShape()[d] == 1)
				lidx[d] = 0;
			else
				lidx[d] = idx[d + offsetL];
		}

		for (size_t d = 0; d < right.Dim(); d++)
		{
			if (right.GetShape()[d] == 1)
				ridx[d] = 0;
			else
				ridx[d] = idx[d + offsetR];
		}

		At(idx) -= right.At(ridx);
	});
}



inline Tensor4D operator*(const Tensor4D& left, const Tensor4D& right)
{
	if (left.GetShape() == right.GetShape())
	{
		Tensor4D result(left.GetShape());

		result.ForEachIndex([&](const std::vector<size_t>& idx)
		{
			result.At(idx) = left.At(idx) * right.At(idx);
		});

		return result;
	}


	if (left.Dim() < 2 || right.Dim() < 2)
		throw std::runtime_error("matmul requires tensors with dim >= 2");

	size_t M = left.GetShape()[4 - 2];
	size_t K = left.GetShape()[4 - 1];

	size_t K2 = right.GetShape()[4 - 2];
	size_t N  = right.GetShape()[4 - 1];

	if (K != K2)
		throw std::runtime_error("matmul inner dimension mismatch");

	// batch dimensions must match

	for (size_t i = 0; i < 4 - 2; i++)
	{
		if (left.GetShape()[i] != right.GetShape()[i])
			throw std::runtime_error("batch dimensions mismatch");
	}

	// output shape

	auto outShape = left.GetShape();
	outShape[4 - 1] = N;

	Tensor4D result(outShape);

	// iterate batches

	result.ForEachIndex([&](const std::vector<size_t>& outIdx)
	{
		float sum = 0.0f;

		for (size_t k = 0; k < K; k++)
		{
			auto lidx = outIdx;
			auto ridx = outIdx;

			lidx.back() = k;

			ridx[ridx.size() - 2] = k;

			sum += left.At(lidx) * right.At(ridx);
		}

		result.At(outIdx) = sum;
	});

	return result;
}