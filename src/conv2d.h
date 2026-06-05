#pragma once
#include "tensor4d.h"
#include "tensor.h"



class Conv2D
{
public:
	Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size, int stride = 1, int padding = 0)
        : m_Stride(stride), m_Padding(padding)
    {
        m_Weights = Node<Tensor4D>::Create(Tensor4D({ out_channels, in_channels, kernel_size, kernel_size }), "conv_weights");

        m_Biases = Node<Tensor4D>::Create(Tensor4D({ out_channels, 1, 1, 1 }), "conv_bias");
        
        float scale = 1.0f;

        for (size_t i = 0; i < m_Weights->GetValue().GetSize(); i++)
            m_Weights->GetValue().Data()[i] = RandomUniform() * scale;
        m_Biases->GetValue().SetZero();

        m_Weights->SetAsTrainable();
        m_Biases->SetAsTrainable();
    }

    NodePtr<Tensor4D>Forward(const NodePtr<Tensor4D>& Input);
	

private:
	NodePtr<Tensor4D> m_Weights;
	NodePtr<Tensor4D> m_Biases;

    int m_Stride  = 1;
    int m_Padding = 0;
};



NodePtr<Tensor4D> Conv2D::Forward(const NodePtr<Tensor4D>& Input)
{
    return Input->Conv2D(m_Weights, m_Stride, m_Padding) + m_Biases;
}