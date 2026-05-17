#pragma once
#include <initializer_list>
#include <stdexcept>
#include <algorithm>



class VectorType;



class VectorType
{
public:
	VectorType(std::initializer_list<float> init)
		: m_Length(init.size())
	{
		m_pValues = new float[m_Length];

		std::copy(init.begin(), init.end(), m_pValues);
	}

	VectorType(size_t Length) : m_Length(Length)
	{
		m_pValues = new float[m_Length];
		std::fill(m_pValues, m_pValues + m_Length, 0.0f);
	}

	VectorType(VectorType&& other) noexcept
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

	VectorType(const VectorType& other)
		: m_Length(other.m_Length)
	{
		m_pValues = new float[m_Length];
		std::copy(other.m_pValues, other.m_pValues + m_Length, m_pValues);
	}

	~VectorType()
	{
		delete[] m_pValues;
		m_pValues = nullptr;
		m_Length = 0;
	}

	size_t GetLength() const { return m_Length; }

	const float* GetValue() const { return m_pValues; }
	float* SetValue() { return m_pValues; }

	float Sum() const
	{
		float sum = 0.0f;
		for (size_t i = 0; i < m_Length; i++)
			sum += m_pValues[i];
		return sum;
	}

	VectorType& operator=(const VectorType& other)
	{
		if (this == &other) return *this;

		delete[] m_pValues;

		m_Length = other.m_Length;
		m_pValues = new float[m_Length];
		std::copy(other.m_pValues, other.m_pValues + m_Length, m_pValues);

		return *this;
	}

	VectorType Tanh() const
	{
		VectorType out = *this;
		for (size_t i = 0; i < m_Length; i++)
			out.m_pValues[i] = std::tanh(m_pValues[i]);
		return out;
	}

	VectorType ReLU() const
	{
		VectorType out = *this;
		for (size_t i = 0; i < m_Length; i++)
			out.m_pValues[i] = std::fmax(0.0f, m_pValues[i]);
		return out;
	}

	//Derivate of ReLU, not standard sign function
	VectorType Sign() const
	{
		VectorType out = *this;
		for (size_t i = 0; i < m_Length; i++)
			out.m_pValues[i] = (m_pValues[i] > 0.0f ? 1.0f : 0.0f);
		return out;
	}

	VectorType operator+(const VectorType& other) const
	{
		VectorType result{};

		if (m_Length != other.m_Length)
			throw std::runtime_error("Vector size mismatch");

		result.m_Length = m_Length;
		result.m_pValues = new float[m_Length];

		for (size_t i = 0; i < m_Length; i++)
			result.m_pValues[i] = m_pValues[i] + other.m_pValues[i];

		return result;
	}

	[[nodiscard]]
	VectorType ElementwiseAdd(const VectorType& other) const
	{
		VectorType result{};

		if (other.m_Length != 1)
			throw std::runtime_error("Vector size mismatch");

		result.m_Length = m_Length;
		result.m_pValues = new float[m_Length];

		for (size_t i = 0; i < m_Length; i++)
			result.m_pValues[i] = m_pValues[i] + other.m_pValues[0];

		return result;
	}

	void operator+=(const VectorType& rhs)
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

	void operator-=(const VectorType& rhs)
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

	VectorType operator*(const VectorType& other) const
	{
		VectorType result{};

		if (m_Length != other.m_Length)
			throw std::runtime_error("Vector size mismatch");

		result.m_Length = m_Length;
		result.m_pValues = new float[m_Length];

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



inline VectorType operator+(VectorType& left, VectorType& right)
{
	VectorType result{};

	if (left.m_Length != right.m_Length)
		throw std::runtime_error("Vector size mismatch");

	result.m_Length  = left.m_Length;
	result.m_pValues = new float[left.m_Length];

	for (size_t i = 0; i < left.m_Length; i++)
		result.m_pValues[i] = left.m_pValues[i] + right.m_pValues[i];

	return result;
}



inline VectorType operator-(VectorType& left, VectorType& right)
{
	VectorType result{};

	if (left.m_Length != right.m_Length)
		throw std::runtime_error("Vector size mismatch");

	result.m_Length  = left.m_Length;
	result.m_pValues = new float[left.m_Length];

	for (size_t i = 0; i < left.m_Length; i++)
		result.m_pValues[i] = left.m_pValues[i] - right.m_pValues[i];

	return result;
}



inline VectorType operator-(float lhs, const VectorType& rhs)
{
	VectorType out = rhs;
	for (size_t i = 0; i < rhs.GetLength(); i++)
		out.m_pValues[i] = lhs - rhs.m_pValues[i];
	return out;
}



inline VectorType operator*(float lhs, const VectorType& rhs)
{
	VectorType out = rhs;
	for (size_t i = 0; i < rhs.GetLength(); i++)
		out.m_pValues[i] = lhs * rhs.m_pValues[i];
	return out;
}