#pragma once
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <algorithm>
#include <vector>



class Matrix;



class Matrix
{
public:
	Matrix() = default;

	Matrix(std::initializer_list<std::initializer_list<float>> init)
		: m_Rows(init.size()), m_Columns(init.begin()->size())
	{
		m_pValues = new float[m_Rows * m_Columns];

		size_t rows = 0;
		for (const auto& row : init)
		{
			if (row.size() != m_Columns)
				throw std::runtime_error("Initializer rows must have equal column size");

			size_t col = 0;
			for (const auto& v : row)
			{
				m_pValues[rows * m_Columns + col] = v;
				col++;
			}
			rows++;
		}
	}

	/*Matrix(const std::vector<float>& Vec)
		: m_Length(Vec.size())
	{
		m_pValues = new float[m_Length];
		std::copy(Vec.begin(), Vec.end(), m_pValues);
	}*/

	Matrix(size_t Rows, size_t Columns) : m_Rows(Rows), m_Columns(Columns)
	{
		m_pValues = new float[m_Rows * m_Columns];
		std::fill(m_pValues, m_pValues + m_Rows * m_Columns, 0.0f);
	}

	Matrix(const Matrix& other)
		: m_Rows(other.m_Rows), m_Columns(other.m_Columns)
	{
		m_pValues = new float[m_Rows * m_Columns];
		std::copy(other.m_pValues, other.m_pValues + m_Rows * m_Columns, m_pValues);
	}

	Matrix(Matrix&& other) noexcept
	{
		m_pValues = other.m_pValues;
		m_Rows    = other.m_Rows;
		m_Columns = other.m_Columns;
		other.m_pValues = nullptr;
		other.m_Rows = 0;
		other.m_Columns = 0;
	}

	Matrix& operator=(const Matrix& other)
	{
		if (this == &other) return *this;

		delete[] m_pValues;

		m_Rows = other.m_Rows;
		m_Columns = other.m_Columns;
		m_pValues = new float[m_Rows * m_Columns];
		std::copy(other.m_pValues, other.m_pValues + m_Rows * m_Columns, m_pValues);

		return *this;
	}

	Matrix& operator=(Matrix&& other) noexcept
	{
		if (this == &other)
			return *this;

		delete[] m_pValues;

		m_pValues = other.m_pValues;
		m_Rows = other.m_Rows;
		m_Columns = other.m_Columns;

		other.m_pValues = nullptr;
		other.m_Rows = 0;
		other.m_Columns = 0;

		return *this;
	}

	void SetLength(size_t Length)
	{
		delete[] m_pValues;
		m_Rows = Length;
		m_Columns = 1;
		m_pValues = new float[Length];
		std::fill(m_pValues, m_pValues + Length, 0.0f);
	}

	void SetZero()
	{
		std::fill(m_pValues, m_pValues + m_Rows * m_Columns, 0.0f);
	}

	/*void SetOne()
	{
		std::fill(m_pValues, m_pValues + m_Rows * m_Columns, 1.0f);
	}*/

	void SetOne()
	{
		std::fill(m_pValues, m_pValues + m_Rows * m_Columns, 0.0f);

		size_t minDim = std::min(m_Rows, m_Columns);

		for (size_t i = 0; i < minDim; ++i)
			m_pValues[i * m_Columns + i] = 1.0f;
	}

	~Matrix()
	{
		delete[] m_pValues;
		m_pValues = nullptr;
		m_Rows = 0;
		m_Columns = 0;
	}

	size_t GetLength() const { return m_Rows * m_Columns; }

	const float* GetValue() const { return m_pValues; }
	float* SetValue() { return m_pValues; }

	size_t GetRows() const { return m_Rows; }
	size_t GetColumns() const { return m_Columns; }

	float& At(size_t row, size_t col)
	{
		return m_pValues[row * m_Columns + col];
	}

	float At(size_t row, size_t col) const
	{
		return m_pValues[row * m_Columns + col];
	}

	Matrix Transpose() const
	{
		Matrix result(m_Columns, m_Rows);

		for (size_t row = 0; row < m_Rows; ++row)
		{
			for (size_t col = 0; col < m_Columns; ++col)
				result.m_pValues[col * result.m_Columns + row] = m_pValues[row * m_Columns + col];
		}

		result.Print();

		return result;
	}

	Matrix Sum() const
	{
		float sum = 0.0f;
		for (size_t i = 0; i < m_Rows * m_Columns; i++)
			sum += m_pValues[i];
		Matrix result({ { sum } });
		return result;
	}

	Matrix Tanh() const
	{
		Matrix out = *this;
		for (size_t i = 0; i < m_Rows * m_Columns; i++)
			out.m_pValues[i] = std::tanh(m_pValues[i]);
		return out;
	}

	Matrix ReLU() const
	{
		Matrix out = *this;
		for (size_t i = 0; i < m_Rows * m_Columns; i++)
			out.m_pValues[i] = std::fmax(0.0f, m_pValues[i]);
		return out;
	}

	//Derivate of ReLU
	Matrix Heaviside() const
	{
		Matrix out = *this;
		for (size_t i = 0; i < m_Rows* m_Columns; i++)
			out.m_pValues[i] = (m_pValues[i] > 0.0f ? 1.0f : 0.0f);
		return out;
	}

	/*[[nodiscard]]
	Matrix ElementwiseAdd(const Matrix& other) const
	{
		return *this + other;
		/*if (m_Rows != other.m_Rows)
			throw std::runtime_error("Vector size mismatch");

		Matrix result(1, 1);

		//for (size_t i = 0; i < m_Length; i++)
			//result.m_pValues[i] = m_pValues[i] + other.m_pValues[0];

		return result;
	}*/

	void operator+=(const Matrix& rhs)
	{
		if ( (m_Rows != rhs.m_Rows || m_Columns != rhs.m_Columns) && (rhs.m_Rows != 1 && rhs.m_Columns != 1) )
			throw std::runtime_error("Matrix operator += size mismatch");

		if (rhs.m_Rows == 1 && rhs.m_Columns == 1)
		{
			for (size_t i = 0; i < m_Rows * m_Columns; i++)
				m_pValues[i] += rhs.m_pValues[0];
			return;
		}

		for (size_t i = 0; i < m_Rows * m_Columns; i++)
			m_pValues[i] += rhs.m_pValues[i];
	}

	void operator+=(float rhs)
	{
		for (size_t i = 0; i < m_Rows * m_Columns; i++)
			m_pValues[i] += rhs;
	}

	void operator-=(const Matrix& rhs)
	{
		if (m_Rows != rhs.m_Rows || m_Columns != rhs.m_Columns)
			throw std::runtime_error("Matrix operator -= size mismatch");

		for (size_t i = 0; i < m_Rows * m_Columns; i++)
			m_pValues[i] -= rhs.m_pValues[i];
	}

	void operator-=(float rhs)
	{
		for (size_t i = 0; i < m_Rows * m_Columns; i++)
			m_pValues[i] -= rhs;
	}

	void operator*=(float rhs)
	{
		for (size_t i = 0; i < m_Rows * m_Columns; i++)
			m_pValues[i] *= rhs;
	}

	void operator/=(float rhs)
	{
		for (size_t i = 0; i < m_Rows * m_Columns; i++)
			m_pValues[i] /= rhs;
	}

	Matrix ElementwiseMul(const Matrix& rhs) const
	{
		if (m_Rows != rhs.m_Rows || m_Columns != rhs.m_Columns)
			throw std::runtime_error("Matrix elementwise multiplication size mismatch");

		Matrix result(m_Rows, m_Columns);

		for (size_t i = 0; i < m_Rows * m_Columns; i++)
			result.m_pValues[i] = m_pValues[i] * rhs.m_pValues[i];

		return result;
	}

	Matrix operator*(const Matrix& rhs) const
	{
		if (m_Rows == 1 && m_Columns == 1)
		{
			Matrix result(rhs.m_Rows, rhs.m_Columns);

			for (size_t i = 0; i < rhs.m_Rows * rhs.m_Columns; i++)
				result.m_pValues[i] = m_pValues[0] * rhs.m_pValues[i];

			return result;
		}

		else if (rhs.m_Rows == 1 && rhs.m_Columns == 1)
		{
			Matrix result(m_Rows, m_Columns);

			for (size_t i = 0; i < m_Rows * m_Columns; i++)
				result.m_pValues[i] = m_pValues[i] * rhs.m_pValues[0];

			return result;
		}

		//Normal matrix multiplication
		else if (m_Columns == rhs.m_Rows)
		{
			Matrix result(m_Rows, rhs.m_Columns);
			this->Print();
			rhs.Print();

			for (size_t row = 0; row < m_Rows; row++)
			{
				for (size_t col = 0; col < rhs.m_Columns; col++)
				{
					float sum = 0.0f;

					for (size_t k = 0; k < m_Columns; k++)
						sum += m_pValues[row * m_Columns + k] * rhs.m_pValues[k * rhs.m_Columns + col];

					result.m_pValues[row * rhs.m_Columns + col] = sum;
				}
			}

			result.Print();

			return result;
		}

		throw std::runtime_error("Matrix multiplication size mismatch");
	}

	float operator[](size_t Index) const
	{
		if (Index >= m_Rows * m_Columns)
			throw std::out_of_range("Matrix index out of range");
		return m_pValues[Index];
	}

	void Print() const
	{
		for (size_t row = 0; row < m_Rows; ++row)
		{
			for (size_t col = 0; col < m_Columns; ++col)
				printf("%.2f ", m_pValues[row * m_Columns + col]);
			printf("\n");
		}
	}

private:
	float* m_pValues = nullptr;
	size_t m_Rows    = 0;
	size_t m_Columns = 0;
};



inline Matrix operator+(const Matrix& left, const Matrix& right)
{
	if (left.GetRows() != right.GetRows() || left.GetColumns() != right.GetColumns())
		throw std::runtime_error("Matrix operator + size mismatch");

	Matrix result(left.GetRows(), left.GetColumns());

	const float* l = left.GetValue();
	const float* r = right.GetValue();
	float* out = result.SetValue();

	for (size_t i = 0; i < left.GetRows() * left.GetColumns(); i++)
		out[i] = l[i] + r[i];

	return result;
}



inline Matrix operator-(const Matrix& left, const Matrix& right)
{
	if (left.GetRows() != right.GetRows() || left.GetColumns() != right.GetColumns())
		throw std::runtime_error("Matrix operator - size mismatch");

	Matrix result(left.GetRows(), left.GetColumns());

	const float* l = left.GetValue();
	const float* r = right.GetValue();
	float* out = result.SetValue();

	for (size_t i = 0; i < left.GetRows() * left.GetColumns(); i++)
		out[i] = l[i] - r[i];

	return result;
}



inline Matrix operator-(float lhs, const Matrix& rhs)
{
	Matrix result = rhs;
	float* out = result.SetValue();
	for (size_t i = 0; i < rhs.GetLength(); i++)
		out[i] = lhs - out[i];
	return result;
}



inline Matrix operator*(float lhs, const Matrix& rhs)
{
	Matrix result = rhs;
	float* out = result.SetValue();
	for (size_t i = 0; i < rhs.GetLength(); i++)
		out[i] = lhs * rhs[i];
	return result;
}