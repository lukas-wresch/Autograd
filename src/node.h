#pragma once
#include <stdexcept>
#include <memory>
#include <vector>
#include <unordered_set>
#include "scalar.h"
#include "vector.h"



enum class Operator
{
    Undefined,
    Add,
    Subtract,
    Multiply,

    Sum,
    ElementwiseAdd,
    ElementwiseMul,
	Pack, // List of nodes -> vector. [x0, x1, x2] -> [(x0, x1, x2)]

    Tanh,
    ReLU,

    Softmax,
};



template<typename T>
class Node;


template<typename T>
using NodePtr = std::shared_ptr<Node<T>>;



template<typename T>
class Node : public std::enable_shared_from_this<Node<T>>
{
public:
    Node() {}

    Node(T value, const std::string& Label = "") : value(value), grad(value), label(Label)
    {}

    Node(size_t Length1, size_t Length2, const std::string& Label = "") : value(Length1, Length2), grad(Length1, Length2), label(Label)
    {}

    T& GetValue() { return value; }
    void SetValue(const T& v){ value = v; }

    T& GetGradient() { return grad; }

    bool IsTrainable() const { return trainable; }
    void SetAsTrainable(bool Trainable = true) { trainable = Trainable; }

    static NodePtr<T> Create(T Value, const std::string& Label = "") { return std::make_shared<Node<T>>(Value, Label); }
    static NodePtr<T> Create() { return std::make_shared<Node<T>>(); }
    static NodePtr<T> CreateWithSize(size_t Length, const std::string& Label = "") { return std::make_shared<Node<T>>(Length, Label); }
    static NodePtr<T> CreateWithSize(size_t Length1, size_t Length2, const std::string& Label = "") { return std::make_shared<Node<T>>(Length1, Length2, Label); }

    NodePtr<T> Sum();
	NodePtr<T> ElementwiseAdd(const NodePtr<T>& other);
    NodePtr<T> ElementwiseMul(const NodePtr<T>& other);
    void Pack(const std::vector<NodePtr<T>>& List);
    NodePtr<T> Tanh();
    NodePtr<T> ReLU();
    NodePtr<T> Softmax();

    void Backwards();

    std::vector<NodePtr<T>> CollectParams();

    const std::string& GetLabel() const { return label;  }
    void SetLabel(const std::string& Label) { label = Label; }

    void Print() const
    {
        if (!label.empty())
            printf("%s: ", label.c_str());
        printf("Value: ");
        for (size_t i = 0; i < value.GetLength(); i++)
            printf("%.4f ", value.GetValue()[i]);
        printf("\nGradient: ");
        for (size_t i = 0; i < grad.GetLength(); i++)
            printf("%.4f ", grad.GetValue()[i]);
        printf("\n");
	}


    T value{};
    T grad{};

    bool trainable = false;

    Operator op = Operator::Undefined;

    NodePtr<T> left  = nullptr;
    NodePtr<T> right = nullptr;
    std::vector<NodePtr<T>> inputs;

    std::string label;


//private:
    void _Backwards(std::unordered_set<Node<T>*>& Visited) const;
    void _ZeroGrads();
    void _CollectParams(std::vector<NodePtr<T>>& out, std::unordered_set<NodePtr<T>>& visited);
};



template<typename T>
void Node<T>::_Backwards(std::unordered_set<Node<T>*>& Visited) const
{
    if (Visited.find(const_cast<Node<T>*>(this)) != Visited.end())
        return;

    Visited.insert(const_cast<Node<T>*>(this));

    if (op == Operator::Undefined)
        return;

    switch (op)
    {
    case Operator::Add:
        left->grad  += grad;
        right->grad += grad;
        break;
    case Operator::Subtract:
        left->grad  += grad;
        right->grad -= grad;
        break;
    case Operator::Multiply:
        left->grad  += grad * right->value.Transpose();
        right->grad += left->value.Transpose() * grad;
        break;
    case Operator::Sum:
        //left->grad = left->grad.ElementwiseAdd(grad);
        left->grad += grad;
        break;
    case Operator::ElementwiseAdd:
        left->grad  += grad;
        right->grad += grad.Sum();
        break;
    case Operator::ElementwiseMul:
        left->grad  += right->value.ElementwiseMul(grad);
        right->grad += left->value.ElementwiseMul(grad);
        break;
    case Operator::Pack:
        for (size_t i = 0; i < inputs.size(); i++)
            inputs[i]->grad += grad.GetValue()[i];
        break;
    case Operator::Tanh:
        left->grad += (1.0f - value.ElementwiseMul(value)).ElementwiseMul(grad);//tanh' = 1 - tanh^2
        break;
    case Operator::ReLU:
		left->grad += value.Heaviside().ElementwiseMul(grad);//ReLU' = 1 if x > 0 else 0
        break;
    default:
        throw std::runtime_error("Unsupported Operation");
        break;
    }

    if (left)
        left->_Backwards(Visited);
    if (right)
        right->_Backwards(Visited);
    for (const auto& input : inputs)
		input->_Backwards(Visited);
}



template<typename T>
void Node<T>::_ZeroGrads()
{
    grad.SetZero();

    if (left)
        left->_ZeroGrads();
    if (right)
        right->_ZeroGrads();
    for (const auto& input : inputs)
		input->_ZeroGrads();
}



template<typename T>
void Node<T>::Backwards()
{
    _ZeroGrads();
    grad.SetOne();
    std::unordered_set<Node<T>*> visited;
    _Backwards(visited);
}



template<typename T>
void Node<T>::_CollectParams(std::vector<NodePtr<T>>& out, std::unordered_set<NodePtr<T>>& visited)
{
    if (visited.find(this->shared_from_this()) != visited.end())
        return;

    visited.insert(this->shared_from_this());

    if (trainable)
        out.push_back(this->shared_from_this());

    if (left)
        left->_CollectParams( out, visited);
    if (right)
        right->_CollectParams(out, visited);

    for (auto& i : inputs)
        i->_CollectParams(out, visited);
}



template<typename T>
std::vector<NodePtr<T>> Node<T>::CollectParams()
{
    std::vector<NodePtr<T>> out;
    std::unordered_set<NodePtr<T>> visited;

    _CollectParams(out, visited);
    return out;
}



template<typename T>
NodePtr<T> operator+(const NodePtr<T>& left, const NodePtr<T>& right)
{
    auto out = std::make_shared<Node<T>>(left->value + right->value);

    out->op = Operator::Add;
    out->left  = left;
    out->right = right;

    return out;
}



template<typename T>
NodePtr<T> operator-(const NodePtr<T>& left, const NodePtr<T>& right)
{
    auto out = std::make_shared<Node<T>>(left->value - right->value);

    out->op = Operator::Subtract;
    out->left  = left;
    out->right = right;

    return out;
}



template<typename T>
NodePtr<T> operator*(const NodePtr<T>& left, const NodePtr<T>& right)
{
    auto out = std::make_shared<Node<T>>(left->value * right->value);

    out->op = Operator::Multiply;
    out->left  = left;
    out->right = right;

    return out;
}



template<typename T>
NodePtr<T> Node<T>::Sum()
{
    auto out = Node<T>::Create(value.Sum());

    out->grad.SetLength(1);

    out->op    = Operator::Sum;
    out->left  = this->shared_from_this();
    out->right = nullptr;
	return out;
}



template<typename T>
NodePtr<T> Node<T>::ElementwiseAdd(const NodePtr<T>& other)
{
    auto out = std::make_shared<Node<T>>(this->value.ElementwiseAdd(other->value));

    out->op    = Operator::ElementwiseAdd;
    out->left  = this->shared_from_this();
    out->right = other;

    return out;
}



template<typename T>
NodePtr<T> Node<T>::ElementwiseMul(const NodePtr<T>& other)
{
    auto out = std::make_shared<Node<T>>(this->value.ElementwiseMul(other->value));

    out->op = Operator::ElementwiseMul;
    out->left = this->shared_from_this();
    out->right = other;

    return out;
}



template<typename T>
void Node<T>::Pack(const std::vector<NodePtr<T>>& List)
{
	value.SetLength(List.size());
	for (size_t i = 0; i < List.size(); i++)
	    value.SetValue()[i] = List[i]->value.GetValue()[0];

	inputs = List;

    op = Operator::Pack;
}



template<typename T>
NodePtr<T> Node<T>::Tanh()
{
    auto out = std::make_shared<Node<T>>(this->value.Tanh());

    out->op    = Operator::Tanh;
    out->left  = this->shared_from_this();
    out->right = nullptr;
    return out;
}



template<typename T>
NodePtr<T> Node<T>::ReLU()
{
    auto out = std::make_shared<Node<T>>(this->value.ReLU());

    out->op = Operator::ReLU;
    out->left = this->shared_from_this();
    out->right = nullptr;
    return out;
}



template<typename T>
NodePtr<T> Node<T>::Softmax()
{
    auto out = std::make_shared<Node<T>>(this->value.Softmax());

    out->op = Operator::Softmax;
    out->left = this->shared_from_this();
    out->right = nullptr;
    return out;
}