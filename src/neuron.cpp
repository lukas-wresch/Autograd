#include <cstdlib>
#include "neuron.h"



float RandomFloatMinus1To1()
{
    return (std::rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}



Neuron::Neuron(Activation Activation) : m_Activation(Activation)
{
    m_Weight = Node<ScalarType>::Create();
    m_Bias   = Node<ScalarType>::Create();
}



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



Neuron2D::Neuron2D(size_t InputLength, Activation Activation) : m_Activation(Activation)
{
    m_Weight = Node<VectorType>::Create(InputLength);
    m_Bias   = Node<VectorType>::Create(1);

    for (size_t i = 0; i < InputLength; i++)
    {
        m_Weight->value.m_pValues[i] = RandomFloatMinus1To1();
        m_Weight->trainable = true;
    }

    m_Bias->value.m_pValues[0] = RandomFloatMinus1To1();
	m_Bias->trainable = true;
}