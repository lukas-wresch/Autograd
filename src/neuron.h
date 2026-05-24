#pragma once
#include "node.h"



enum class Activation
{
	Identity,
	ReLU,
	Sigmoid,
	Tanh
};



class Neuron
{
public:
	Neuron(Activation Activation) : m_Activation(Activation)
	{
		m_Weight = Node<Scalar>::Create();
		m_Bias   = Node<Scalar>::Create();
	}

	template<typename T>
	NodePtr<T> Forward(const NodePtr<T>& Input) const;


private:
	NodePtr<Scalar> m_Weight;
	NodePtr<Scalar> m_Bias;
	Activation m_Activation = Activation::Identity;
};



class Neuron2D
{
public:
	Neuron2D() = default;
	Neuron2D(size_t InputLength, Activation Activation);

	template<typename T>
	NodePtr<T> Forward(const NodePtr<T>& Input) const;

	auto GetWeight() const { return m_Weight; }
	auto GetBias()   const { return m_Bias; }

	void Print() const
	{
		printf("Weight: ");
		for (size_t i = 0; i < m_Weight->value.GetLength(); i++)
			printf("%.4f ", m_Weight->value.m_pValues[i]);
		printf("\nBias: %.4f\n", m_Bias->value.m_pValues[0]);

		printf("Gradients: ");
		for (size_t i = 0; i < m_Weight->grad.GetLength(); i++)
			printf("%.4f ", m_Weight->grad.m_pValues[i]);
		printf("\nBias Gradient: %.4f\n", m_Bias->grad.m_pValues[0]);
	}


private:
	NodePtr<Vector> m_Weight;
	NodePtr<Vector> m_Bias;
	Activation m_Activation = Activation::Identity;
};



template<typename T>
NodePtr<T> Neuron::Forward(const NodePtr<T>& Input) const
{
	auto ret = m_Weight * Input;
	ret = ret->ElementwiseAdd(m_Bias);

	switch (m_Activation)
	{
		/*case Activation::ReLU:
			z.SetValue(std::max(0.0f, z.GetValue()));
			break;
		case Activation::Sigmoid:
			z.SetValue(1.0f / (1.0f + std::exp(-z.GetValue())));
			break;*/
	case Activation::Tanh:
		ret = ret->Tanh();
		break;
	default:
		break;
	}

	return ret;
}



inline float RandomFloatMinus1To1()
{
	return (std::rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}



Neuron2D::Neuron2D(size_t InputLength, Activation Activation) : m_Activation(Activation)
{
	m_Weight = Node<Vector>::Create(InputLength);
	m_Bias = Node<Vector>::Create(1);

	for (size_t i = 0; i < InputLength; i++)
	{
		m_Weight->value.m_pValues[i] = RandomFloatMinus1To1();
		m_Weight->SetAsTrainable();
	}

	m_Bias->value.m_pValues[0] = RandomFloatMinus1To1();
	m_Bias->SetAsTrainable();
}



template<typename T>
NodePtr<T> Neuron2D::Forward(const NodePtr<T>& Input) const
{
	auto z = m_Weight * Input;
	NodePtr<T> sum = z->Sum();
	NodePtr<T> pre_activation = sum + m_Bias;

	switch (m_Activation)
	{
	case Activation::Identity:
		return pre_activation;
		break;
		/*case Activation::ReLU:
			z.SetValue(std::max(0.0f, z.GetValue()));
			break;
		case Activation::Sigmoid:
			z.SetValue(1.0f / (1.0f + std::exp(-z.GetValue())));
			break;*/
	case Activation::Tanh:
		return pre_activation->Tanh();
		break;
	default:
		throw std::runtime_error("Unsupported activation function");
		break;
	}

	return pre_activation;
}