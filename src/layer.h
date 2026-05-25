#pragma once
#include <vector>
#include "neuron.h"
#include "matrix.h"



template<typename T>
class Layer2D
{
public:
	Layer2D(size_t InputLength, size_t OutputLength, Activation Activation);

	~Layer2D() { delete[] m_pNeurons; }

	NodePtr<T> Forward(const NodePtr<T>& Input) const;

	Neuron2D* GetNeuron(size_t Index) { return &m_pNeurons[Index]; }
	NodePtr<Vector> GetWeight(size_t Index) { return m_pNeurons[Index].GetWeight(); }
	NodePtr<Vector> GetBias(  size_t Index) { return m_pNeurons[Index].GetBias();   }

	size_t GetOutputLength() const { return m_OutputLength; }

	void Print() const
	{
		for (size_t i = 0; i < m_OutputLength; i++)
		{
			printf("Neuron %zu:\n", i);
			m_pNeurons[i].Print();
			printf("\n");
		}
	}

private:
	Neuron2D* m_pNeurons;
	size_t m_OutputLength = 0;
};



class Layer3D
{
public:
	Layer3D(size_t InputLength, size_t OutputLength, Activation Activation) : m_Activation(Activation), m_OutputLength(OutputLength)
	{
		m_Weights = Node<Matrix>::CreateWithSize(OutputLength, InputLength);
		m_Biases  = Node<Matrix>::CreateWithSize(OutputLength, 1);

		float scale = std::sqrt(1.0f / InputLength); // Xavier Initialization

		for (size_t i = 0; i < m_Weights->GetValue().GetLength(); i++)
			m_Weights->GetValue().SetValue()[i] = RandomFloatMinus1To1() * scale;
		for (size_t i = 0; i < m_Biases->GetValue().GetLength(); i++)
			m_Biases->GetValue().SetValue()[i]  = RandomFloatMinus1To1() * scale;

		m_Weights->SetAsTrainable();
		m_Biases->SetAsTrainable();
		m_Weights->SetLabel("W");
		m_Biases->SetLabel("B");
	}

	NodePtr<Matrix> Forward(const NodePtr<Matrix>& Input) const;

	NodePtr<Matrix> GetWeights() { return m_Weights; }
	NodePtr<Matrix> GetBiases()  { return m_Biases;  }

	size_t GetOutputLength() const { return m_Weights->GetValue().GetRows(); }

private:
	NodePtr<Matrix> m_Weights;
	NodePtr<Matrix> m_Biases;
	Activation m_Activation;
	size_t m_OutputLength;
};



template<typename T>
Layer2D<T>::Layer2D(size_t InputLength, size_t OutputLength, Activation Activation)
{
	m_pNeurons = new Neuron2D[OutputLength];
	m_OutputLength = OutputLength;

	for (size_t i = 0; i < OutputLength; i++)
		m_pNeurons[i] = Neuron2D(InputLength, Activation);
}



template<typename T>
NodePtr<T> Layer2D<T>::Forward(const NodePtr<T>& Input) const
{
	NodePtr<T> out = Node<T>::CreateWithSize(m_OutputLength);
	std::vector<NodePtr<T>> outputs(m_OutputLength);

	for (size_t i = 0; i < m_OutputLength; i++)
		outputs[i] = m_pNeurons[i].Forward(Input);

	out->Pack(outputs);
	return out;
}



NodePtr<Matrix> Layer3D::Forward(const NodePtr<Matrix>& Input) const
{
	auto pre_activation = m_Weights * Input + m_Biases;

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
		throw std::runtime_error("Layer3D: Unsupported activation function");
		break;
	}

	return pre_activation;
}