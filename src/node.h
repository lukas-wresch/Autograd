#pragma once
#include <stdexcept>
#include <memory>
#include <vector>
#include <unordered_set>
#include "vector.h"
#include "tensor4d.h"
#include "kernels.h"



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
    CrossEntropy,
    Softmax_CrossEntropy,

    Conv2D,
    MaxPool2D,
    Flatten,

    BatchNorm,
    BatchNorm2D,

    GlobalAveragePool2D,
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

    Node(const T& value, const std::string& Label = "") : value(value), grad(value.Clone()), label(Label)
    {}

    Node(size_t Length1, size_t Length2, const std::string& Label = "") : value(Length1, Length2), grad(Length1, Length2), label(Label)
    {}

    T& GetValue() { return value; }
    void SetValue(const T& v){ value = v; }

    T& GetGradient() { return grad; }

    bool IsTrainable() const { return trainable; }
    void SetAsTrainable(bool Trainable = true) { trainable = Trainable; }
    bool HasWeightDecay() const { return weight_decay; }
    void SetWeightDecay(bool DecayEnabled = true) { weight_decay = DecayEnabled; }

    static NodePtr<T> Create(const T& Value, const std::string& Label = "") { return std::make_shared<Node<T>>(Value, Label); }
    static NodePtr<T> Create() { return std::make_shared<Node<T>>(); }
    static NodePtr<T> CreateWithSize(size_t Length, const std::string& Label = "") { return std::make_shared<Node<T>>(Length, Label); }
    static NodePtr<T> CreateWithSize(size_t Length1, size_t Length2, const std::string& Label = "") { return std::make_shared<Node<T>>(Length1, Length2, Label); }
    static NodePtr<T> Create(std::initializer_list<float> init, const std::string& Label = "") { return std::make_shared<Node<T>>(init, Label); }
    static NodePtr<T> Create(std::initializer_list<std::initializer_list<float>> init, const std::string& Label = "") { return std::make_shared<Node<T>>(init, Label); }

    NodePtr<T> Sum();
	NodePtr<T> ElementwiseAdd(const NodePtr<T>& other);
    NodePtr<T> ElementwiseMul(const NodePtr<T>& other);
    void Pack(const std::vector<NodePtr<T>>& List);
    NodePtr<T> Tanh();
    NodePtr<T> ReLU();
    NodePtr<T> Softmax();
    NodePtr<T> CrossEntropy(NodePtr<T> Target);
    NodePtr<T> Softmax_CrossEntropy(NodePtr<T> Target);
    NodePtr<T> Conv2D(NodePtr<T> Kernel, int Stride = 1, int Padding = 0);
    NodePtr<T> MaxPool2D(int KernelSize, int Stride);
    NodePtr<T> Flatten();
    NodePtr<T> BatchNorm();
    NodePtr<T> BatchNorm2D();
    NodePtr<T> GlobalAveragePool2D();

    void Backwards();

    std::vector<NodePtr<T>> CollectParams();

    const std::string& GetLabel() const { return label;  }
    void SetLabel(const std::string& Label) { label = Label; }
    size_t GetSize() const { return value.GetSize(); }

    void Print() const
    {
        if (!label.empty())
            printf("%s: ", label.c_str());
        printf("Value: ");
        for (size_t i = 0; i < value.GetSize(); i++)
            printf("%.4f ", value.Data()[i]);
        printf("\nGradient: ");
        for (size_t i = 0; i < grad.GetSize(); i++)
            printf("%.4f ", grad.Data()[i]);
        printf("\n\n");
	}


    T value{};
    T grad{};

    bool trainable = false;
    bool weight_decay = false;

    Operator op = Operator::Undefined;

    // Convolution
    int stride  = 1;
    int padding = 0;
    int kernel_size = 1;

    // Flatten
    size_t B = 0;
    size_t C = 0;
    size_t H = 0;
    size_t W = 0;

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
        if constexpr (std::is_same_v<T, Tensor4D>)
        {
            left->grad  += grad % right->value.Transpose();
            right->grad += left->value.Transpose() % grad;
        }
        else
        {
            left->grad  += grad * right->value.Transpose();
            right->grad += left->value.Transpose() * grad;
        }
        break;
    case Operator::Sum:
        left->grad += grad;
        break;
    case Operator::ElementwiseAdd:
        left->grad += grad;
        right->grad += grad.Sum();
        break;
    case Operator::ElementwiseMul:
        if constexpr (std::is_same_v<T, Tensor4D>)
        {
            left->grad += right->value * grad;
            right->grad += left->value * grad;
        }
        else
        {
            left->grad += right->value.ElementwiseMul(grad);
            right->grad += left->value.ElementwiseMul(grad);
        }
        break;
    case Operator::Pack:
        if constexpr (std::is_same_v<T, Tensor4D>)
            throw std::runtime_error("Unsupported Operation");
        else
        {
            for (size_t i = 0; i < inputs.size(); i++)
                inputs[i]->grad += grad.GetValue()[i];
        }
        break;
    case Operator::Tanh:
        if constexpr (std::is_same_v<T, Tensor4D>)
            left->grad += (1.0f - value * value) * grad;//tanh' = 1 - tanh^2
        else
            left->grad += (1.0f - value.ElementwiseMul(value)).ElementwiseMul(grad);//tanh' = 1 - tanh^2
        break;
    case Operator::ReLU:
        if constexpr (std::is_same_v<T, Tensor4D>)
            left->grad += value.Heaviside() * grad;//ReLU' = 1 if x > 0 else 0
        else
            left->grad += value.Heaviside().ElementwiseMul(grad);//ReLU' = 1 if x > 0 else 0
        break;
    case Operator::Softmax:
    {
        {
            float dot = 0.0f;

            for (size_t i = 0; i < grad.GetSize(); i++)
                dot += grad[i] * value[i];

            for (size_t i = 0; i < grad.GetSize(); i++)
                left->grad.Data()[i] += value[i] * (grad[i] - dot);
        }
        break;
    }
    case Operator::CrossEntropy:
    {
        {
            T& probs = left->value;
            T& grad_probs = left->grad;

            const int target = (int)right->value.Data()[0];

            grad_probs.Data()[target] += (-1.0f / probs[target]) * grad[0];
        }
        break;
    }
    case Operator::Softmax_CrossEntropy:
    {
        {
            // logits are stored in "left"
            T& logits = left->value;
            T& grad_logits = left->grad;

            const int target = (int)right->value.Data()[0];

            // forward softmax recomputation
            // (oder cached probabilities!)
            float max_val = logits.Max(); // numerical stability

            float sum = 0.0f;
            std::vector<float> probs(logits.GetSize());

            for (size_t i = 0; i < logits.GetSize(); i++)
            {
                probs[i] = std::exp(logits[i] - max_val);
                sum += probs[i];
            }

            for (float& p : probs)
                p /= sum;

            // backward: dL/dlogits = p - y
            for (size_t i = 0; i < logits.GetSize(); i++)
            {
                float y = (i == (size_t)target) ? 1.0f : 0.0f;
                grad_logits.Data()[i] += (probs[i] - y) * grad[0];
            }
        }
        break;
    }
    case Operator::MaxPool2D:
        if constexpr (std::is_same_v<T, Tensor4D>)
            Kernels::MaxPool2D_Backward(left->grad, left->value, grad, right->value, this->kernel_size, this->stride);
        else
            throw std::runtime_error("Unsupported Operation");
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

    out->op    = Operator::Add;
    out->left  = left;
    out->right = right;

    return out;
}



template<typename T>
NodePtr<T> operator-(const NodePtr<T>& left, const NodePtr<T>& right)
{
    auto out = std::make_shared<Node<T>>(left->value - right->value);

    out->op    = Operator::Subtract;
    out->left  = left;
    out->right = right;

    return out;
}



template<typename T>
NodePtr<T> operator*(const NodePtr<T>& left, const NodePtr<T>& right)
{
    if constexpr (std::is_same_v<T, Tensor4D>)
    {
        auto out = std::make_shared<Node<T>>(left->value % right->value);
        out->op = Operator::Multiply;
        out->left = left;
        out->right = right;

        return out;
    }
    else
    {
        auto out = std::make_shared<Node<T>>(left->value * right->value);
        out->op = Operator::Multiply;
        out->left = left;
        out->right = right;

        return out;
    }
}



template<typename T>
NodePtr<T> operator%(const NodePtr<T>& left, const NodePtr<T>& right)
{
    if constexpr (std::is_same_v<T, Tensor4D>)
    {
        auto out = std::make_shared<Node<T>>(left->value % right->value);
        out->op = Operator::Multiply;
        out->left = left;
        out->right = right;

        return out;
    }
    else
    {
        auto out = std::make_shared<Node<T>>(left->value * right->value);
        out->op = Operator::Multiply;
        out->left = left;
        out->right = right;

        return out;
    }
}



template<typename T>
NodePtr<T> Node<T>::Sum()
{
    auto out = Node<T>::Create(value.Sum());

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
    if constexpr (std::is_same_v<T, Tensor4D>)
    {
        auto out   = std::make_shared<Node<T>>(this->value * other->value);
        out->op    = Operator::ElementwiseMul;
        out->left  = this->shared_from_this();
        out->right = other;

        return out;
    }
    else
    {
        auto out   = std::make_shared<Node<T>>(this->value.ElementwiseMul(other->value));
        out->op    = Operator::ElementwiseMul;
        out->left  = this->shared_from_this();
        out->right = other;

        return out;
    }
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



template<typename T>
NodePtr<T> Node<T>::Softmax_CrossEntropy(NodePtr<T> Target)
{
    auto out = std::make_shared<Node<T>>(this->value.Softmax().CrossEntropy(Target->GetValue()));

    out->op = Operator::Softmax_CrossEntropy;
    out->left = this->shared_from_this();
    out->right = Target;
    return out;
}



template<typename T>
NodePtr<T> Node<T>::Conv2D(NodePtr<T> Kernel, int Stride, int Padding)
{
    size_t N = this->GetValue().GetShape()[0];
    size_t H = this->GetValue().GetShape()[2];
    size_t W = this->GetValue().GetShape()[3];

    size_t kernel_size = Kernel->GetValue().GetShape()[2];
    size_t out_channels = Kernel->GetValue().GetShape()[0];
    size_t H_out = (H + 2 * Padding - kernel_size) / Stride + 1;
    size_t W_out = (W + 2 * Padding - kernel_size) / Stride + 1;

    auto tensor = Tensor4D({ N, out_channels, H_out, W_out });

    Kernels::Conv2D_Forward(tensor, this->value, Kernel->value, Stride, Padding);

    auto out = std::make_shared<Node<T>>(tensor);

    out->op      = Operator::Conv2D;
    out->left    = this->shared_from_this();
    out->right   = Kernel;
	out->stride  = Stride;
    out->padding = Padding;
    return out;
}



template<typename T>
NodePtr<T> Node<T>::MaxPool2D(int KernelSize, int Stride)
{
    auto out = std::make_shared<Node<T>>(Kernels::MaxPool2D_Forward(this->value, KernelSize, Stride));

    out->op   = Operator::MaxPool2D;
    out->left = this->shared_from_this();
    out->kernel_size = KernelSize;
    out->stride      = Stride;
    return out;
}



template<typename T>
NodePtr<T> Node<T>::Flatten()
{
    size_t C = value.GetShape()[1];
    size_t H = value.GetShape()[2];
    size_t W = value.GetShape()[3];

    auto out = std::make_shared<Node<T>>(this->value.Reshape({ this->value.GetBatches(), 1, C*H*W, 1 } ));

    out->op = Operator::Flatten;
    out->left = this->shared_from_this();
    out->B = this->value.GetBatches();
	out->C = C;
	out->H = H;
    out->W = W;
    return out;
}



template<typename T>
NodePtr<T> Node<T>::BatchNorm()
{
    auto out = std::make_shared<Node<T>>(this->value.BatchNorm());

    out->op = Operator::BatchNorm;
    out->left = this->shared_from_this();
    return out;
}



template<typename T>
NodePtr<T> Node<T>::BatchNorm2D()
{
    auto out = std::make_shared<Node<T>>(this->value.BatchNorm2D());

    out->op = Operator::BatchNorm2D;
    out->left = this->shared_from_this();
    return out;
}



template<typename T>
NodePtr<T> Node<T>::GlobalAveragePool2D()
{
    auto out = std::make_shared<Node<T>>(this->value.GlobalAveragePool2D());

    out->op = Operator::GlobalAveragePool2D;
    out->left = this->shared_from_this();
    return out;
}
