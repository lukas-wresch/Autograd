#pragma once
#include "tensor4d.h"
#include "node.h"
#include "neuron.h"



class BatchNormLayer
{
public:
	BatchNormLayer(size_t Features, const std::string& Label = "")
    {
        m_Gamma = Node<Tensor4D>::Create(Tensor4D({ 1, 1, Features, 1 }));
        m_Beta  = Node<Tensor4D>::Create(Tensor4D({ 1, 1, Features, 1 }));


        m_Gamma->GetValue().SetOne();
        m_Beta->GetValue().SetZero();

        m_Gamma->SetAsTrainable();
        m_Beta->SetAsTrainable();
        m_Gamma->SetLabel(Label + "Gain");
        m_Beta->SetLabel(Label  + "B");
    }

    NodePtr<Tensor4D> Forward(const NodePtr<Tensor4D>& Input);
	

private:
	NodePtr<Tensor4D> m_Gamma;
	NodePtr<Tensor4D> m_Beta;
};



NodePtr<Tensor4D> BatchNormLayer::Forward(const NodePtr<Tensor4D>& Input)
{
	return m_Gamma->ElementwiseMul(Input->BatchNorm()) + m_Beta;
}



class BatchNorm2DLayer
{
public:
    BatchNorm2DLayer(size_t Depth, const std::string& Label = "")
    {
        m_Gamma = Node<Tensor4D>::Create(Tensor4D({ 1, Depth, 1, 1 }));
        m_Beta  = Node<Tensor4D>::Create(Tensor4D({ 1, Depth, 1, 1 }));


        m_Gamma->GetValue().SetOne();
        m_Beta->GetValue().SetZero();

        m_Gamma->SetAsTrainable();
        m_Beta->SetAsTrainable();
        m_Gamma->SetLabel(Label + "Gamma");
        m_Beta->SetLabel(Label  + "Beta");
    }

    NodePtr<Tensor4D> Forward(const NodePtr<Tensor4D>& Input);


private:
    NodePtr<Tensor4D> m_Gamma;
    NodePtr<Tensor4D> m_Beta;
};



NodePtr<Tensor4D> BatchNorm2DLayer::Forward(const NodePtr<Tensor4D>& Input)
{
    return m_Gamma->ElementwiseMul(Input->BatchNorm2D()) + m_Beta;
}