#pragma once
#include "tensor4d.h"
#include "tensor4d.h"



class Conv2D
{
public:
	Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size, int stride = 1, int padding = 0, Activation Activation = Activation::Identity)
        : m_Stride(stride), m_Padding(padding), m_Activation(Activation)
    {
        m_Weights = Node<Tensor4D>::Create(Tensor4D({ out_channels, in_channels, kernel_size, kernel_size }), "conv_weights");

        m_Biases = Node<Tensor4D>::Create(Tensor4D({ 1, out_channels, 1, 1 }), "conv_bias");

		float fan_in = static_cast<float>(in_channels * kernel_size * kernel_size);
		float fan_out = static_cast<float>(out_channels * kernel_size * kernel_size);

		float gain = std::sqrt(6.0f / (fan_in + fan_out));

        for (size_t i = 0; i < m_Weights->GetValue().GetSize(); i++)
            m_Weights->GetValue().Data()[i] = RandomNormal() * gain;
        m_Biases->GetValue().SetZero();

        m_Weights->SetAsTrainable();
        m_Biases->SetAsTrainable();
    }

    NodePtr<Tensor4D> Conv(const NodePtr<Tensor4D>& Input);
	

private:
	NodePtr<Tensor4D> m_Weights;
	NodePtr<Tensor4D> m_Biases;

    Activation m_Activation;

    int m_Stride  = 1;
    int m_Padding = 0;
};



NodePtr<Tensor4D> Conv2D::Conv(const NodePtr<Tensor4D>& Input)
{
    auto conv_output = Input->Conv2D(m_Weights, m_Stride, m_Padding) + m_Biases;

	switch (m_Activation)
	{
	case Activation::Identity:
		return conv_output;
		break;
	case Activation::ReLU:
		return conv_output->ReLU();
		break;
		//case Activation::Sigmoid:
			//z.SetValue(1.0f / (1.0f + std::exp(-z.GetValue())));
			//break;
	case Activation::Tanh:
		return conv_output->Tanh();
		break;
	default:
		throw std::runtime_error("Layer3D: Unsupported activation function");
		break;
	}
}