#pragma once
#include <vector>
#include "node.h"
#include "tape_recorder.h"



template<typename T>
class SGD
{
public:
    SGD(float lr, float Momentum = 0.0f) : m_LR(lr), m_Momentum(Momentum) {}

    template<typename T>
    void SetTrainableParams(const std::vector<NodePtr<T>>& params)
    {
		this->params = params;
    }

    float GetLearningRate() const { return m_LR; }
    void  SetLearningRate(float lr) { m_LR = lr; }

    size_t GetNumberOfParameters() const
    {
        size_t size = 0;
        for (auto g : grads)
            size += g->GetLength();
        return size;
    }

    void SetTrainableParams(TapeRecorder<T>& Tape)
    {
        for (size_t i = 0; i < Tape.GetNumberOfValues(); i++)
        {
            if (Tape.IsTrainable(i))
            {
                values.push_back(Tape.SetValue(i));
                grads.push_back(Tape.SetGradient(i));
                velocity.push_back(*Tape.SetValue(i));
                velocity.back().SetZero();
            }
        }
    }

    void Step(float Scale = 1.0f)
    {
        if (params.empty())
        {
            //throw std::runtime_error("No parameters to train");

            for (size_t i = 0; i < values.size(); i++)
            {
                velocity[i] = m_Momentum * velocity[i] - m_LR * Scale * (*grads[i]);
                *values[i] += velocity[i];
                //*values[i] -= lr * Scale * (*grads[i]);
            }
        }


        else
        {
            for (auto& p : params)
                p->GetValue() -= m_LR * Scale * p->GetGradient();
        }
    }

private:
    float m_LR = 0.0f;
    float m_Momentum = 0.0f;

    std::vector<NodePtr<T>> params;

    std::vector<T> velocity;

    std::vector<T*> values;
    std::vector<T*> grads;
};