#pragma once
#include "tensor4d.h"
#include "node.h"
#include "neuron.h"



class BatchNormLayer
{
public:
	BatchNormLayer(size_t Features, const std::string& Label = "")
    {
        m_Weights = Node<Tensor4D>::Create(Tensor4D({ 1, 1, Features, 1 }));

        m_Biases = Node<Tensor4D>::Create(Tensor4D({ 1, Features, 1, 1 }));

        float gain = std::sqrt(6.0f / (Features + Features));

        for (size_t i = 0; i < m_Weights->GetValue().GetSize(); i++)
            m_Weights->GetValue().Data()[i] = RandomNormal() * gain;
        m_Biases->GetValue().SetZero();

        m_Weights->SetAsTrainable();
        m_Biases->SetAsTrainable();
		m_Weights->SetLabel(Label + "Gain");
		m_Biases->SetLabel(Label + "B");
    }

    NodePtr<Tensor4D> Conv(const NodePtr<Tensor4D>& Input);
	

private:
	NodePtr<Tensor4D> m_Weights;
	NodePtr<Tensor4D> m_Biases;

    int m_Stride  = 1;
    int m_Padding = 0;
};



NodePtr<Tensor4D> BatchNormLayer::Forward(const NodePtr<Tensor4D>& Input)
{
	return m_Weights * Input->BatchNorm() + m_Biases;
}