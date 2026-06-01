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



	Tensor(Tensor& T) : m_Storage(new Storage(T.m_Storage->GetSize(), T.m_Storage->GetData())), m_Shape(T.m_Shape), m_Strides(T.m_Strides), m_Offset(T.m_Offset)
	{}

	Tensor(const Tensor& T) : m_Storage(new Storage(T.m_Storage->GetSize(), T.m_Storage->GetData())), m_Shape(T.m_Shape), m_Strides(T.m_Strides), m_Offset(T.m_Offset)
	{}

	void operator=(const Tensor& T)
	{
		m_Storage = T.m_Storage;
		m_Shape   = T.m_Shape;
		m_Strides = T.m_Strides;
		m_Offset  = T.m_Offset;
	}


	Tensor(Tensor&& T) noexcept : m_Storage(T.m_Storage), m_Shape(T.m_Shape), m_Strides(T.m_Strides), m_Offset(T.m_Offset)
	{
		T.m_Storage = nullptr;
		T.m_Shape   = {};
		T.m_Strides = {};
		T.m_Offset  = 0;
	}

	Tensor(std::vector<size_t> Shape) : m_Shape(Shape)
	{
		size_t size = 1;
		for (auto s : Shape)
			size *= s;

		m_Storage = new Storage(size);
		m_Strides = ComputeStrides(Shape);
	}

	Tensor(const std::vector<size_t>& Shape, const std::vector<float>& Data): m_Storage(new Storage(Data)), m_Shape(Shape)
	{
		size_t expected = 1;
		for (auto s : m_Shape)
			expected *= s;

		if (expected != Data.size())
			throw std::runtime_error("Shape does not match data size");

		m_Strides = ComputeStrides(m_Shape);
	}

	Tensor(const std::vector<size_t>& Shape, const float* Data) : m_Shape(Shape)
	{
		size_t size = 1;
		for (auto s : m_Shape)
			size *= s;

		m_Storage = new Storage(size, Data);

		m_Strides = ComputeStrides(m_Shape);
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
		return Tensor(m_Storage, new_shape, m_Strides);
	}

	Tensor ViewColumn(size_t col) const
	{
		if (m_Shape.size() != 2)
			throw std::runtime_error("ViewColumn only valid for 2D tensors");
		return Tensor(m_Storage, { m_Shape[0], 1 }, { m_Strides[0], 0 }, m_Offset + col);
	}

	const std::vector<size_t>& Shape() const { return m_Shape; }
	const std::vector<size_t>& Strides() const { return m_Strides; }
	size_t Offset() const { return m_Offset; }

	size_t Size() const
	{
		size_t s = 1;

		for (size_t v : m_Shape)
			s *= v;

		return s;
	}

	size_t GetSize() const
	{
		size_t s = 1;

		for (size_t v : m_Shape)
			s *= v;

		return s;
	}

	void SetZero()
	{
		std::fill(Data(), Data() + Size(), 0.0f);
	}

	void SetOne()
	{
		std::fill(Data(), Data() + Size(), 1.0f);
	}

	float* Data()
	{
		if (m_Storage)
			return m_Storage->SetData() + m_Offset;
		return nullptr;
	}

	const float* Data() const
	{
		if (m_Storage)
			return m_Storage->GetData() + m_Offset;
		return nullptr;
	}

	inline size_t Offset(const std::vector<size_t>& idx) const
	{
		size_t offset = m_Offset;

		for (size_t i = 0; i < m_Shape.size(); i++)
			offset += idx[i] * m_Strides[i];

		return offset;
	}

	std::vector<size_t> BroadcastStrides(const std::vector<size_t>& outShape) const
	{
		std::vector<size_t> result(outShape.size(), 0);

		size_t ndimDiff = outShape.size() - m_Shape.size();

		for (size_t i = 0; i < m_Shape.size(); i++)
		{
			if (m_Shape[i] == 1)
				result[i + ndimDiff] = 0;
			else
				result[i + ndimDiff] = m_Strides[i];
		}

		return result;
	}

	float At(const std::vector<size_t>& idx) const
	{
		if (idx.size() != m_Shape.size())
			throw std::runtime_error("dimension mismatch");

		size_t offset = m_Offset;

		size_t i = 0;
		for (auto it = idx.begin(); it != idx.end(); ++it, ++i)
		{
			if (*it >= m_Shape[i])
				throw std::out_of_range("index out of bounds");

			offset += (*it) * m_Strides[i];
		}

		return m_Storage->GetData()[offset];
	}

	float& At(const std::vector<size_t>& idx)
	{
		if (idx.size() != m_Shape.size())
			throw std::runtime_error("dimension mismatch");

		size_t offset = m_Offset;

		size_t i = 0;
		for (auto it = idx.begin(); it != idx.end(); ++it, ++i)
		{
			if (*it >= m_Shape[i])
				throw std::out_of_range("index out of bounds");

			offset += (*it) * m_Strides[i];
		}

		return m_Storage->SetData()[offset];
	}

	float At(size_t Row, size_t Column) const
	{
		if (m_Shape.size() == 1 && Row == 0 && Column == 0)
			return At({ 0 });

		if (m_Shape.size() != 2)
			throw std::runtime_error("Only matrix type supported");

		return At({ Row, Column });
	}

	float& At(size_t Row, size_t Column)
	{
		if (m_Shape.size() == 1 && Row == 0 && Column == 0)
			return At({ 0 });

		if (m_Shape.size() != 2)
			throw std::runtime_error("Only matrix type supported");

		return At({ Row, Column });
	}

	void Transpose()
	{
		if (m_Shape.size() != 2)
			throw std::runtime_error("Transpose only valid for 2D tensors");

		std::swap(m_Shape[0], m_Shape[1]);
		std::swap(m_Strides[0], m_Strides[1]);
	}

	size_t GetRows() const
	{
		if (m_Shape.size() != 2)
			throw std::runtime_error("SetRow only valid for 2D tensors");
		return m_Shape[0];
	}

	size_t GetColumns() const
	{
		if (m_Shape.size() != 2)
			throw std::runtime_error("GetColumns only valid for 2D tensors");
		return m_Shape[1];
	}

	std::string Shape2String() const
	{
		std::string out;
		for (auto s : m_Shape)
		{
			if (!out.empty())
				out += "x";
			out += std::to_string(s);
		}			
		return out;
	}

	void Print() const
	{
		std::cout << "Tensor(" << Shape2String() << ")\n";
		std::cout << "Offset: " << m_Offset << "\n";
		std::cout << "Strides: ";

		for (size_t i = 0; i < m_Strides.size(); i++)
		{
			std::cout << m_Strides[i];
			if (i + 1 < m_Strides.size()) std::cout << ", ";
		}
		std::cout << "\nValues:\n";

		// 1D Tensor
		if (m_Shape.size() == 1)
		{
			std::cout << "[ ";
			for (size_t i = 0; i < m_Shape[0]; i++)
			{
				std::cout << std::fixed << std::setprecision(4) << At({ i }) << " ";
			}
			std::cout << "]\n";
			return;
		}

		// 2D Tensor
		if (m_Shape.size() == 2)
		{
			for (size_t i = 0; i < m_Shape[0]; i++)
			{
				std::cout << "[ ";
				for (size_t j = 0; j < m_Shape[1]; j++)
				{
					std::cout << std::fixed << std::setprecision(4) << At({ i, j }) << " ";
				}
				std::cout << "]\n";
			}
			return;
		}

		// ND fallback (flattened view)
		std::cout << "[flattened]\n[ ";
		for (size_t i = 0; i < Size(); i++)
		{
			std::cout << std::fixed << std::setprecision(4) << Data()[i] << " ";
		}
		std::cout << "]\n";
	}

	void SetRow(size_t row, std::initializer_list<float> vec)
	{
		if (m_Shape.size() != 2)
			throw std::runtime_error("SetRow only valid for 2D tensors");

		size_t cols = m_Shape[1];

		if (vec.size() != cols)
			throw std::runtime_error("Row size mismatch");

		float* data = m_Storage->SetData();

		for (size_t col = 0; col < cols; col++)
		{
			size_t idx = m_Offset + row * m_Strides[0] + col * m_Strides[1];
			data[idx] = *(vec.begin() + row);
		}
	}

	void SetColumn(size_t col, std::initializer_list<float> vec)
	{
		if (m_Shape.size() != 2)
			throw std::runtime_error("SetRow only valid for 2D tensors");

		size_t rows = m_Shape[0];

		if (vec.size() != rows)
			throw std::runtime_error("Column size mismatch");

		float* data = m_Storage->SetData();

		size_t i = 0;
		for (auto v : vec)
		{
			size_t idx = m_Offset + i * m_Strides[0] + col * m_Strides[1];
			data[idx] = v;
			i++;
		}
	}



	Tensor Sum() const
	{
		float sum = 0.0f;
		for (size_t i = 0; i < Size(); i++)
			sum += Data()[i];
		return Tensor({ 1 }, { sum });
	}

	float Max() const
	{
		float max = Data()[0];
		for (size_t i = 1; i < Size(); i++)
		{
			if (Data()[i] > max)
				max = Data()[i];
		}
		return max;
	}

	size_t ArgMax() const
	{
		if (m_Shape.size() != 1)
			throw std::runtime_error("Only vector type supported");

		size_t index = 0;
		float max = m_Storage->GetData()[0];
		for (size_t i = 1; i < m_Storage->GetSize(); i++)
		{
			if (m_Storage->GetData()[i] > max)
			{
				max = m_Storage->GetData()[i];
				index = i;
			}
		}
		return index;
	}

	Tensor Tanh() const
	{
		Tensor out(m_Shape);
		for (size_t i = 0; i < Size(); i++)
			out.Data()[i] = std::tanh(Data()[i]);
		return out;
	}

	Tensor ReLU() const
	{
		Tensor out(m_Shape);
		for (size_t i = 0; i < Size(); i++)
			out.Data()[i] = std::fmax(0.0f, Data()[i]);
		return out;
	}

	//Derivate of ReLU
	Tensor Heaviside() const
	{
		Tensor out(m_Shape);
		for (size_t i = 0; i < Size(); i++)
			out.Data()[i] = (Data()[i] > 0.0f ? 1.0f : 0.0f);
		return out;
	}

	inline Tensor ElementwiseMul(const Tensor& rhs) const;

	Tensor Softmax() const
	{
		float max_value = Max();

		Tensor out(Shape());

		float sum = 0.0f;
		for (size_t i = 0; i < Size(); i++)
		{
			out.Data()[i] = std::exp(Data()[i] - max_value);
			sum += Data()[i];
		}

		for (size_t i = 0; i < Size(); i++)
			out.Data()[i] /= sum;

		return out;
	}

	Tensor CrossEntropy(const Tensor& Target) const
	{
		if (m_Shape.size() > 2)
			throw std::runtime_error("Only vector/matrix type supported");

		const float epsilon = 0.00001f;
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

			return Tensor({ 1 }, { loss / GetColumns() });
		}

		//Target is on-hot vector (Dense)
		if (GetRows() != Target.GetRows() || GetColumns() != Target.GetColumns())
			throw std::runtime_error("Matrix CrossEntropy size mismatch");

		for (size_t i = 0; i < GetRows(); i++)
			for (size_t j = 0; j < GetColumns(); j++)
				loss -= Target.At({ i, j }) * std::log(At({ i, j }) + epsilon);

		return Tensor({ 1 }, { loss / GetColumns() });
	}


private:
	Tensor(Storage* Storage, const std::vector<size_t>& Shape) : m_Storage(Storage), m_Shape(Shape)
	{
		m_Strides = ComputeStrides(Shape);
	}

	Tensor(Storage* Storage, const std::vector<size_t>& Shape, const std::vector<size_t>& Strides, size_t Offset = 0)
		: m_Storage(Storage), m_Shape(Shape), m_Strides(Strides), m_Offset(Offset)
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



	Storage* m_Storage = nullptr;
	std::vector<size_t> m_Shape;
	std::vector<size_t> m_Strides;
	size_t m_Offset = 0;
};



inline void ForEachIndex(const Tensor& t, std::function<void(const std::vector<size_t>&)> fn)
{
	std::vector<size_t> idx(t.Shape().size(), 0);

	size_t total = t.Size();

	for (size_t i = 0; i < total; i++)
	{
		size_t tmp = i;

		for (int d = (int)t.Shape().size() - 1; d >= 0; d--)
		{
			idx[d] = tmp % t.Shape()[d];
			tmp /= t.Shape()[d];
		}

		fn(idx);
	}
}



inline std::vector<size_t> BroadcastShape(const std::vector<size_t>& a, const std::vector<size_t>& b)
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



inline bool IsBroadcastCompatible(const std::vector<size_t>& ShapeA, const std::vector<size_t>& ShapeB)
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



inline Tensor operator+(const Tensor& left, const Tensor& right)
{
	if (!IsBroadcastCompatible(left.Shape(), right.Shape()))
		throw std::runtime_error("broadcast mismatch");

	auto outShape = BroadcastShape(left.Shape(), right.Shape());

	Tensor result(outShape);

	ForEachIndex(result, [&](const std::vector<size_t>& idx)
	{
		std::vector<size_t> lidx(left.Shape().size());
		std::vector<size_t> ridx(right.Shape().size());

		// map output index -> left/right broadcast index

		int offsetL = (int)outShape.size() - (int)left.Shape().size();
		int offsetR = (int)outShape.size() - (int)right.Shape().size();

		for (size_t d = 0; d < left.Shape().size(); d++)
		{
			if (left.Shape()[d] == 1)
				lidx[d] = 0;
			else
				lidx[d] = idx[d + offsetL];
		}

		for (size_t d = 0; d < right.Shape().size(); d++)
		{
			if (right.Shape()[d] == 1)
				ridx[d] = 0;
			else
				ridx[d] = idx[d + offsetR];
		}

		result.At(idx) = left.At(lidx) + right.At(ridx);
	});

	return result;
}



inline void operator+=(Tensor& left, const Tensor& right)
{
	if (!IsBroadcastCompatible(left.Shape(), right.Shape()))
		throw std::runtime_error("broadcast mismatch");

	auto outShape = BroadcastShape(left.Shape(), right.Shape());

	ForEachIndex(outShape, [&](const std::vector<size_t>& idx)
	{
		std::vector<size_t> lidx(left.Shape().size());
		std::vector<size_t> ridx(right.Shape().size());

		// map output index -> left/right broadcast index

		int offsetL = (int)outShape.size() - (int)left.Shape().size();
		int offsetR = (int)outShape.size() - (int)right.Shape().size();

		for (size_t d = 0; d < left.Shape().size(); d++)
		{
			if (left.Shape()[d] == 1)
				lidx[d] = 0;
			else
				lidx[d] = idx[d + offsetL];
		}

		for (size_t d = 0; d < right.Shape().size(); d++)
		{
			if (right.Shape()[d] == 1)
				ridx[d] = 0;
			else
				ridx[d] = idx[d + offsetR];
		}

		left.At(lidx) += right.At(ridx);
	});
}



inline Tensor operator-(const Tensor& left, const Tensor& right)
{
	if (!IsBroadcastCompatible(left.Shape(), right.Shape()))
		throw std::runtime_error("broadcast mismatch");

	auto outShape = BroadcastShape(left.Shape(), right.Shape());

	Tensor result(outShape);

	ForEachIndex(result, [&](const std::vector<size_t>& idx)
	{
		std::vector<size_t> lidx(left.Shape().size());
		std::vector<size_t> ridx(right.Shape().size());

		// map output index -> left/right broadcast index

		int offsetL = (int)outShape.size() - (int)left.Shape().size();
		int offsetR = (int)outShape.size() - (int)right.Shape().size();

		for (size_t d = 0; d < left.Shape().size(); d++)
		{
			if (left.Shape()[d] == 1)
				lidx[d] = 0;
			else
				lidx[d] = idx[d + offsetL];
		}

		for (size_t d = 0; d < right.Shape().size(); d++)
		{
			if (right.Shape()[d] == 1)
				ridx[d] = 0;
			else
				ridx[d] = idx[d + offsetR];
		}

		result.At(idx) = left.At(lidx) - right.At(ridx);
	});

	return result;
}



inline void operator-=(Tensor& left, const Tensor& right)
{
	if (!IsBroadcastCompatible(left.Shape(), right.Shape()))
		throw std::runtime_error("broadcast mismatch");

	auto outShape = BroadcastShape(left.Shape(), right.Shape());

	ForEachIndex(outShape, [&](const std::vector<size_t>& idx)
	{
		std::vector<size_t> lidx(left.Shape().size());
		std::vector<size_t> ridx(right.Shape().size());

		// map output index -> left/right broadcast index

		int offsetL = (int)outShape.size() - (int)left.Shape().size();
		int offsetR = (int)outShape.size() - (int)right.Shape().size();

		for (size_t d = 0; d < left.Shape().size(); d++)
		{
			if (left.Shape()[d] == 1)
				lidx[d] = 0;
			else
				lidx[d] = idx[d + offsetL];
		}

		for (size_t d = 0; d < right.Shape().size(); d++)
		{
			if (right.Shape()[d] == 1)
				ridx[d] = 0;
			else
				ridx[d] = idx[d + offsetR];
		}

		left.At(lidx) -= right.At(ridx);
	});
}



inline Tensor Tensor::ElementwiseMul(const Tensor& rhs) const
{
	if (Shape() != rhs.Shape())
		throw std::runtime_error("Tensor elementwise multiplication shape mismatch");

	Tensor result(Shape());

	ForEachIndex(rhs, [&](const std::vector<size_t>& idx)
	{
		result.At(idx) = At(idx) * rhs.At(idx);
	});

	return result;
}



inline Tensor operator*(const Tensor& left, const Tensor& right)
{
	//if (left.Shape() != right.Shape())
		//throw std::runtime_error("Tensor operator * shape mismatch");

	if (left.Shape() == right.Shape())
	{
		Tensor result(left.Shape());

		ForEachIndex(left, [&](const std::vector<size_t>& idx)
		{
			result.At(idx) = left.At(idx) * right.At(idx);
		});

		return result;
	}



	if (left.Shape().size() < 2 || right.Shape().size() < 2)
		throw std::runtime_error("matmul requires tensors with dim >= 2");

	size_t M = left.Shape()[left.Shape().size() - 2];
	size_t K = left.Shape()[left.Shape().size() - 1];

	size_t K2 = right.Shape()[right.Shape().size() - 2];
	size_t N  = right.Shape()[right.Shape().size() - 1];

	if (K != K2)
		throw std::runtime_error("matmul inner dimension mismatch");

	// batch dimensions must match

	if (left.Shape().size() != right.Shape().size())
		throw std::runtime_error("batched matmul dims mismatch");

	for (size_t i = 0; i < left.Shape().size() - 2; i++)
	{
		if (left.Shape()[i] != right.Shape()[i])
			throw std::runtime_error("batch dimensions mismatch");
	}

	// output shape

	std::vector<size_t> outShape = left.Shape();
	outShape[outShape.size() - 1] = N;

	Tensor result(outShape);

	// iterate batches

	ForEachIndex(result, [&](const std::vector<size_t>& outIdx)
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



inline Tensor operator*(float lhs, const Tensor& rhs)
{
	Tensor result(rhs.Shape());

	ForEachIndex(rhs, [&](const std::vector<size_t>& idx)
	{
		result.At(idx) = lhs * rhs.At(idx);
	});

	return result;
}



inline Tensor Matrix2Tensor(const Matrix& In)
{
	Tensor out({ In.GetRows(), In.GetColumns() }, In.GetValue());
	return out;
}



inline Matrix Tensor2Matrix(const Tensor& In)
{
	Matrix out(In.Shape()[0], In.Shape()[1]);	
	std::copy(In.Data(), In.Data() + In.Size(), out.SetValue());
	return out;
}