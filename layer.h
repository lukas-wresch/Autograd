#pragma once
#include <vector>
#include "neuron.h"



template<typename T>
class Layer2D
{
public:
	Layer2D(size_t InputLength, size_t OutputLength, Activation Activation);

	~Layer2D() { delete[] m_pNeurons; }

	NodePtr<T> Forward(const NodePtr<T>& Input) const;

	Neuron2D* GetNeuron(size_t Index) { return &m_pNeurons[Index]; }
	NodePtr<VectorType> GetWeight(size_t Index) { return m_pNeurons[Index].GetWeight(); }
	NodePtr<VectorType> GetBias(  size_t Index) { return m_pNeurons[Index].GetBias();   }

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