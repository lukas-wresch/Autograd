#pragma once
#include <vector>
#include "node.h"



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

    void Step()
    {
        if (params.empty())
            throw std::runtime_error("No parameters to train");


        for (auto& p : params)
            p->GetValue() -= lr * p->GetGradient();
    }

private:
    float lr;

    std::vector<NodePtr<T>> params;
};