#pragma once
#include <vector>
#include "node.h"
#include "tape_recorder.h"



template<typename T>
class SGD
{
public:
    SGD(float lr) : lr(lr) {}

    template<typename T>
    void SetTrainableParams(const std::vector<NodePtr<T>>& params)
    {
		this->params = params;
    }

    template<typename T>
    void SetTrainableParams(TapeRecorder<T>& Tape)
    {
        for (size_t i = 0; i < Tape.GetNumberOfValues(); i++)
        {
            if (Tape.IsTrainable(i))
            {
                values.push_back(Tape.SetValue(i));
                grads.push_back(Tape.SetGradient(i));
            }
        }
    }

    void Step(float Scale = 1.0f)
    {
        if (params.empty())
        {
            //throw std::runtime_error("No parameters to train");

            for (size_t i = 0; i < values.size(); i++)
                *values[i] -= lr * Scale * (*grads[i]);
        }


        else
        {
            for (auto& p : params)
                p->GetValue() -= lr * Scale * p->GetGradient();
        }
    }

private:
    float lr;

    std::vector<NodePtr<T>> params;

    std::vector<T*> values;
    std::vector<T*> grads;
};