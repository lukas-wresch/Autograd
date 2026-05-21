#pragma once
#include <initializer_list>
#include <stdexcept>
#include <algorithm>
#include <vector>



class Vector
{
public:
	Vector() = default;

	Vector(std::initializer_list<float> init)
		: m_Length(init.size())
	{
		m_pValues = new float[m_Length];

		std::copy(init.begin(), init.end(), m_pValues);
	}

	Vector(const std::vector<float>& Vec)
		: m_Length(Vec.size())
	{
		m_pValues = new float[m_Length];
		std::copy(Vec.begin(), Vec.end(), m_pValues);
	}

	Vector(size_t Length) : m_Length(Length)
	{
		m_pValues = new float[m_Length];
		std::fill(m_pValues, m_pValues + m_Length, 0.0f);
	}

	Vector(const Vector& other)
		: m_Length(other.m_Length)
	{
		m_pValues = new float[m_Length];
		std::copy(other.m_pValues, other.m_pValues + m_Length, m_pValues);
	}

	Vector(Vector&& other) noexcept
	{
		m_pValues = other.m_pValues;
		m_Length  = other.m_Length;
		other.m_pValues = nullptr;
		other.m_Length = 0;
	}

	void SetLength(size_t Length)
	{
		delete[] m_pValues;
		m_Length = Length;
		m_pValues = new float[Length];
		std::fill(m_pValues, m_pValues + Length, 0.0f);
	}

	void SetZero()
	{
		std::fill(m_pValues, m_pValues + m_Length, 0.0f);
	}

	void SetOne()
	{
		std::fill(m_pValues, m_pValues + m_Length, 1.0f);
	}

	~Vector()
	{
		delete[] m_pValues;
		m_pValues = nullptr;
		m_Length = 0;
	}

	size_t GetLength() const { return m_Length; }

	const float* GetValue() const { return m_pValues; }
	float* SetValue() { return m_pValues; }

	Vector Sum() const
	{
		float sum = 0.0f;
		for (size_t i = 0; i < m_Length; i++)
			sum += m_pValues[i];
		Vector result({ sum });
		return result;
	}

	Vector& operator=(const Vector& other)
	{
		if (this == &other) return *this;

		delete[] m_pValues;

		m_Length = other.m_Length;
		m_pValues = new float[m_Length];
		std::copy(other.m_pValues, other.m_pValues + m_Length, m_pValues);

		return *this;
	}

	Vector& operator=(Vector&& other) noexcept
	{
		if (this == &other)
			return *this;

		delete[] m_pValues;

		m_pValues = other.m_pValues;
		m_Length  = other.m_Length;

		other.m_pValues = nullptr;
		other.m_Length = 0;

		return *this;
	}

	Vector Tanh() const
	{
		Vector out = *this;
		for (size_t i = 0; i < m_Length; i++)
			out.m_pValues[i] = std::tanh(m_pValues[i]);
		return out;
	}

	Vector ReLU() const
	{
		Vector out = *this;
		for (size_t i = 0; i < m_Length; i++)
			out.m_pValues[i] = std::fmax(0.0f, m_pValues[i]);
		return out;
	}

	//Derivate of ReLU
	Vector Heaviside() const
	{
		Vector out = *this;
		for (size_t i = 0; i < m_Length; i++)
			out.m_pValues[i] = (m_pValues[i] > 0.0f ? 1.0f : 0.0f);
		return out;
	}

	[[nodiscard]]
	Vector ElementwiseAdd(const Vector& other) const
	{
		if (other.m_Length != 1)
			throw std::runtime_error("Vector size mismatch");

		Vector result(m_Length);

		for (size_t i = 0; i < m_Length; i++)
			result.m_pValues[i] = m_pValues[i] + other.m_pValues[0];

		return result;
	}

	void operator+=(const Vector& rhs)
	{
		if (m_Length != rhs.m_Length)
			throw std::runtime_error("Vector size mismatch");

		for (size_t i = 0; i < m_Length; i++)
			m_pValues[i] += rhs.m_pValues[i];
	}

	void operator+=(float rhs)
	{
		for (size_t i = 0; i < m_Length; i++)
			m_pValues[i] += rhs;
	}

	void operator-=(const Vector& rhs)
	{
		if (m_Length != rhs.m_Length)
			throw std::runtime_error("Vector size mismatch");

		for (size_t i = 0; i < m_Length; i++)
			m_pValues[i] -= rhs.m_pValues[i];
	}

	void operator-=(float rhs)
	{
		for (size_t i = 0; i < m_Length; i++)
			m_pValues[i] -= rhs;
	}

	void operator*=(float rhs)
	{
		for (size_t i = 0; i < m_Length; i++)
			m_pValues[i] *= rhs;
	}

	void operator/=(float rhs)
	{
		for (size_t i = 0; i < m_Length; i++)
			m_pValues[i] /= rhs;
	}

	Vector operator*(const Vector& other) const
	{
		if (m_Length != other.m_Length)
			throw std::runtime_error("Vector operator *: size mismatch");

		Vector result(m_Length);

		for (size_t i = 0; i < m_Length; i++)
			result.m_pValues[i] = m_pValues[i] * other.m_pValues[i];

		return result;
	}

	float operator[](size_t Index) const
	{
		if (Index >= m_Length)
			throw std::out_of_range("Vector index out of range");
		return m_pValues[Index];
	}

	void Print() const
	{
		for (size_t i = 0; i < m_Length; i++)
			printf("%.4f ", m_pValues[i]);
		printf("\n");
	}

//private:
	float* m_pValues = nullptr;
	size_t m_Length  = 0;
};



inline Vector operator+(const Vector& left, const Vector& right)
{
	if (left.m_Length != right.m_Length)
		throw std::runtime_error("Vector operator +: Size mismatch");

	Vector result(left.m_Length);

	for (size_t i = 0; i < left.m_Length; i++)
		result.m_pValues[i] = left.m_pValues[i] + right.m_pValues[i];

	return result;
}



inline Vector operator-(const Vector& left, const Vector& right)
{
	if (left.m_Length != right.m_Length)
		throw std::runtime_error("Vector operator -: Size mismatch");

	Vector result(left.m_Length);

	for (size_t i = 0; i < left.m_Length; i++)
		result.m_pValues[i] = left.m_pValues[i] - right.m_pValues[i];

	return result;
}



inline Vector operator-(float lhs, const Vector& rhs)
{
	Vector out = rhs;
	for (size_t i = 0; i < rhs.GetLength(); i++)
		out.m_pValues[i] = lhs - rhs.m_pValues[i];
	return out;
}



inline Vector operator*(float lhs, const Vector& rhs)
{
	Vector out = rhs;
	for (size_t i = 0; i < rhs.GetLength(); i++)
		out.m_pValues[i] = lhs * rhs.m_pValues[i];
	return out;
}