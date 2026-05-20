#pragma once
#include <stdexcept>
#include <cmath>



class ScalarType
{
public:
	ScalarType() : value(0.0f) {}
	ScalarType(float Value) : value(Value) {}

	void operator=(const ScalarType& rhs)
	{
		value = rhs.value;
	}

	void operator=(float rhs)
	{
		value = rhs;
	}

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

	ScalarType ElementwiseAdd(const ScalarType& rhs) const
	{
		return ScalarType(value + rhs.value);
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

	//Derivate of ReLU
	float Heaviside() const
	{
		return (value > 0.0f ? 1.0f : 0.0f);
	}
	
	void operator+=(const ScalarType& rhs)
	{
		value += rhs.value;
	}

	void operator+=(float rhs)
	{
		value += rhs;
	}

	void operator-=(const ScalarType& rhs)
	{
		value -= rhs.value;
	}

	void operator-=(float rhs)
	{
		value -= rhs;
	}

	void operator*=(const ScalarType& rhs)
	{
		value *= rhs.value;
	}

	ScalarType operator+(const ScalarType& other) const
	{
		return ScalarType(value + other.value);
	}

	ScalarType operator-(const ScalarType& other) const
	{
		return ScalarType(value - other.value);
	}

	ScalarType operator*(const ScalarType& other) const
	{
		return ScalarType(value * other.value);
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



inline ScalarType operator-(float lhs, const ScalarType& rhs)
{
	ScalarType out = lhs - rhs.value;
	return out;
}