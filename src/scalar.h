#pragma once
#include <stdexcept>
#include <cmath>



class Scalar
{
public:
	Scalar() : value(0.0f) {}
	Scalar(float Value) : value(Value) {}

	void operator=(const Scalar& rhs)
	{
		value = rhs.value;
	}

	void operator=(float rhs)
	{
		value = rhs;
	}

	Scalar Clone() const
	{
		return Scalar(*this);
	}

	const float* Data() const { return &value; }
	float* Data() { return &value; }
	const float* GetValue() const { return &value; }
	void   SetValue(float Value) { value = Value; }
	float* SetValue() { return &value; }

	void SetZero()
	{
		value = 0.0f;
	}

	void SetOne()
	{
		value = 1.0f;
	}

	void SetLength(size_t Length)
	{
		if (Length != 1)
			throw std::runtime_error("Scalar size mismatch");
		value = 0.0f;
	}

	size_t GetLength() const { return 1; }
	size_t GetSize() const { return 1; }
	size_t GetRows() const { return 1; }
	size_t GetColumns() const { return 1; }
	size_t GetDepth() const { return 1; }
	size_t GetBatches() const { return 1; }
	std::string Shape2String() const { return std::string("1"); }

	Scalar ElementwiseAdd(const Scalar& rhs) const
	{
		return Scalar(value + rhs.value);
	}

	Scalar Transpose() const
	{
		return *this;
	}

	float Sum() const
	{
		return value;
	}

	float Tanh() const
	{
		return std::tanh(value);
	}

	float ReLU() const
	{
		return std::fmax(0.0f, value);
	}

	float Max() const
	{
		return value;
	}

	//Derivate of ReLU
	Scalar Heaviside() const
	{
		return Scalar(value > 0.0f ? 1.0f : 0.0f);
	}

	Scalar Softmax() const
	{
		return Scalar(1);
	}

	Scalar CrossEntropy(const Scalar& Target) const
	{
		throw std::runtime_error("Scalar CrossEntropy not implemented");
		return Scalar(0.0f);
	}
	
	void operator+=(const Scalar& rhs)
	{
		value += rhs.value;
	}

	void operator+=(float rhs)
	{
		value += rhs;
	}

	void operator-=(const Scalar& rhs)
	{
		value -= rhs.value;
	}

	void operator-=(float rhs)
	{
		value -= rhs;
	}

	void operator*=(const Scalar& rhs)
	{
		value *= rhs.value;
	}

	Scalar operator+(const Scalar& other) const
	{
		return Scalar(value + other.value);
	}

	Scalar operator-(const Scalar& other) const
	{
		return Scalar(value - other.value);
	}

	Scalar ElementwiseMul(const Scalar& other) const
	{
		return Scalar(value * other.value);
	}

	Scalar operator*(const Scalar& other) const
	{
		return Scalar(value * other.value);
	}

	operator float() const { return value; }

	float operator[](size_t Index) const
	{
		if (Index != 0)
			throw std::out_of_range("Scalar index out of range");
		return value;
	}


//private:
	float value = 0.0f;
};



inline Scalar operator-(float lhs, const Scalar& rhs)
{
	Scalar out = lhs - rhs.value;
	return out;
}