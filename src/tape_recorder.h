#pragma once
#include <unordered_map>
#include <string>
#include <stdio.h>
#include "node.h"
#include "tensor4d.h"
#include "kernels.h"



struct TapeEntry
{
    Operator op;

    int a;   // input tensor index
    int b;   // optional
    int c;   // optional, used for cache
    std::vector<int> inputs;

    int out; // output tensor index

    // Convolution
    int stride  = 1;
    int padding = 0;

	// Max Pool
	int kernel_size = 1;

    // Flatten
    size_t B;
    size_t C;
    size_t H;
    size_t W;
};



template<typename T>
class TapeRecorder
{
public:
    TapeRecorder() = default;
    TapeRecorder(const NodePtr<T>& root) { Compile(root); ZeroGradients(); }

    void Compile(const NodePtr<T>& root);

    void Forward();

    void ZeroGradients();

    void Backward();

    const T* GetValue(const std::string& Label) const
    {
        auto it = label_to_id.find(Label);
        if (it != label_to_id.end())
            return &values[it->second];
        return nullptr;
    }

    T* SetValue(const std::string& Label)
    {
        auto it = label_to_id.find(Label);
        if (it != label_to_id.end())
            return &values[it->second];
        return nullptr;
    }


    const T* GetGradient(const std::string& Label) const
    {
        auto it = label_to_id.find(Label);
        if (it != label_to_id.end())
            return &grads[it->second];
        return nullptr;
    }

    T* SetValue(size_t Index)
    {
        return &values[Index];
    }

    T* SetGradient(size_t Index)
    {
        return &grads[Index];
    }

    bool IsTrainable(size_t Index) const
    {
        return trainable[Index];
    }

    size_t GetNumberOfValues() const
    {
        return values.size();
    }

    int AddDataEntry(const T& v, bool Trainable = false)
    {
        values.push_back(v);

        grads.push_back(v.Clone());

        trainable.push_back(Trainable);

        return (int)values.size() - 1;
    }

	void AddOpEntry(Operator op, int a, int b, int out)
	{
        tape.push_back({ op, a, b, -1, {}, out });
	}

    void AddOpEntryConv2D(Operator op, int a, int b, int out, int stride, int padding)
    {
        TapeEntry new_entry({ op, a, b, -1, {}, out, stride, padding });
        new_entry.stride = stride;
        new_entry.padding = padding;
        tape.push_back(new_entry);
    }

    void AddOpEntryMaxPool(Operator op, int a, int b, int out, int KernelSize, int Stride)
    {
        TapeEntry new_entry({ op, a, b, -1, {}, out, Stride, 0 });
        new_entry.kernel_size = KernelSize;
        new_entry.stride = Stride;
        tape.push_back(new_entry);
    }

    void AddOpEntry(Operator op, int a, int out, size_t B, size_t C, size_t H, size_t W)
    {
        TapeEntry new_entry({ op, a, -1, -1, {}, out, 1, 0 });
        new_entry.B = B;
        new_entry.C = C;
        new_entry.H = H;
        new_entry.W = W;
        tape.push_back(new_entry);
    }

    void AddOpEntry(Operator op, const std::vector<int>& inputs, int out)
    {
        tape.push_back({ op, -1, -1, -1, inputs, out });
    }

    void PrintTape() const;

private:
    void Visit(const NodePtr<T>& node, std::unordered_map<NodePtr<T>, int>& node_to_id);

	std::vector<TapeEntry> tape;

    std::vector<T> values;
    std::vector<T> grads;
    std::vector<bool> trainable;

    std::unordered_map<std::string, int> label_to_id;
    std::unordered_map<int, std::string> id_to_label;
};



template<typename T>
void TapeRecorder<T>::Visit(const NodePtr<T>& node, std::unordered_map<NodePtr<T>, int>& node_to_id)
{
    auto GetID = [&](const NodePtr<T>& node) -> int
    {
        auto it = node_to_id.find(node);
        if (it != node_to_id.end())
            return it->second;
        return -1; // Not found
    };

	if (GetID(node) != -1)//Already visited?
		return;

    if (node->left)
        Visit(node->left, node_to_id);
    if (node->right)
        Visit(node->right, node_to_id);
    for (auto& i : node->inputs)
        Visit(i, node_to_id);


    auto node_id = AddDataEntry(node->GetValue(), node->IsTrainable());
    if (!node->GetLabel().empty())
    {
        label_to_id.insert({ node->GetLabel(), node_id });
        id_to_label.insert({ node_id, node->GetLabel() });
    }
	node_to_id.insert({ node, node_id });
    std::vector<int> input_ids;

    switch (node->op)
    {
    case Operator::Undefined:
        break;

    // Two operands
    case Operator::Add:
    case Operator::Subtract:
    case Operator::Multiply:
    case Operator::ElementwiseAdd:
    case Operator::ElementwiseMul:
    case Operator::CrossEntropy:
    case Operator::Softmax_CrossEntropy:
        AddOpEntry(node->op, GetID(node->left), GetID(node->right), node_id);
        break;

	// One Operand
	case Operator::Sum:
    case Operator::Tanh:
    case Operator::ReLU:
    case Operator::Softmax:
        AddOpEntry(node->op, GetID(node->left), -1, node_id);
        break;

    // Convolutions
    case Operator::Conv2D:
        AddOpEntryConv2D(node->op, GetID(node->left), GetID(node->right), node_id, node->stride, node->padding);
        break;

    // Max Pool
    case Operator::MaxPool2D:
    {
        auto cache_id = AddDataEntry(node->GetValue());
        AddOpEntryMaxPool(node->op, GetID(node->left), cache_id, node_id, node->kernel_size, node->stride);
        break;
    }

    // Flatten
    case Operator::Flatten:
        AddOpEntry(node->op, GetID(node->left), node_id, node->B, node->C, node->H, node->W);
        break;

	//N operands
    case Operator::Pack:
		for (const auto& input : node->inputs)
			input_ids.push_back(GetID(input));

		AddOpEntry(node->op, input_ids, node_id);
		break;

	default:
		throw std::runtime_error("Unsupported Operation");
    }
}



template<typename T>
inline void TapeRecorder<T>::Compile(const NodePtr<T>& root)
{
    std::unordered_map<NodePtr<T>, int> node_to_id;
    Visit(root, node_to_id);
}



template<typename T>
void TapeRecorder<T>::Forward()
{
	for (const auto& entry : tape)
	{
        switch (entry.op)
        {
		case Operator::Add:
            values[entry.out] = values[entry.a] + values[entry.b];
			break;
		case Operator::Subtract:
            values[entry.out] = values[entry.a] - values[entry.b];
            break;
        case Operator::Multiply:
            if constexpr (std::is_same_v<T, Tensor>)
                Kernels::MatMul_Forward(values[entry.out], values[entry.a], values[entry.b]);
            else if constexpr (std::is_same_v<T, Tensor4D>)
                values[entry.out] = values[entry.a] % values[entry.b];
            else
                values[entry.out] = values[entry.a] * values[entry.b];
            break;

        case Operator::Sum:
            values[entry.out] = values[entry.a].Sum();
            break;
        case Operator::ElementwiseAdd:
            if constexpr (std::is_same_v<T, Tensor>)
                values[entry.out] = values[entry.a] + values[entry.b];
            else if constexpr (std::is_same_v<T, Tensor4D>)
                values[entry.out] = values[entry.a] + values[entry.b];
            else
                values[entry.out] = values[entry.a].ElementwiseAdd(values[entry.b]);
            break;
        case Operator::ElementwiseMul:
            if constexpr (std::is_same_v<T, Tensor>)
                Kernels::Multiply_Forward(values[entry.out], values[entry.a], values[entry.b]);
            else if constexpr (std::is_same_v<T, Tensor4D>)
            {
                values[entry.out] = values[entry.a] * values[entry.b];
            }
            else
                values[entry.out] = values[entry.a].ElementwiseMul(values[entry.b]);
            break;

        case Operator::Pack:
            if constexpr (std::is_same_v<T, Tensor>)
            {
                throw std::runtime_error("Unsupported Operation");
            }
            else if constexpr (std::is_same_v<T, Tensor4D>)
            {
                throw std::runtime_error("Unsupported Operation");
            }
            else
            {
                values[entry.out].SetLength(entry.inputs.size());//Neccessary???
                for (size_t j = 0; j < entry.inputs.size(); j++)
                    values[entry.out].SetValue()[j] = values[entry.inputs[j]].GetValue()[0];
            }
            break;

        case Operator::Tanh:
            values[entry.out] = values[entry.a].Tanh();
            break;
        case Operator::ReLU:
            values[entry.out] = values[entry.a].ReLU();
            break;
        case Operator::Softmax_CrossEntropy:
            if constexpr (std::is_same_v<T, Tensor>)
            {
                values[entry.out] = values[entry.a].Softmax().CrossEntropy(values[entry.b]);
            }
            else
                values[entry.out] = values[entry.a].Softmax().CrossEntropy(values[entry.b]);
            break;
        case Operator::Conv2D:
            if constexpr (std::is_same_v<T, Tensor4D>)
                values[entry.out] = Kernels::Conv2D_Forward(values[entry.a], values[entry.b], entry.stride, entry.padding);
            else
                throw std::runtime_error("Unsupported Operation");
            break;
        case Operator::MaxPool2D:
            if constexpr (std::is_same_v<T, Tensor4D>)
                values[entry.out] = Kernels::MaxPool2D_Forward(values[entry.a], entry.kernel_size, entry.stride, &values[entry.b]);
            else
                throw std::runtime_error("Unsupported Operation");
            break;
        case Operator::Flatten:
            if constexpr (std::is_same_v<T, Tensor4D>)
            {
                values[entry.out].CopyFrom(values[entry.a]);
                values[entry.out] = values[entry.out].Reshape({ entry.B, 1, entry.C * entry.H * entry.W, 1 });
            }
            else
                throw std::runtime_error("Unsupported Operation");
            break;

        default:
            throw std::runtime_error("Unsupported Operation");
        }
	}
}



template<typename T>
inline void TapeRecorder<T>::ZeroGradients()
{
    for (auto& g : grads)
        g.SetZero();

    grads[grads.size() - 1].SetOne();
}



template<typename T>
inline void TapeRecorder<T>::Backward()
{
    //Zero all the gradients except the trainable parameter. For them gradients should accumulate
    for (size_t i = 0; i < grads.size() - 1; i++)
    {
        if (!trainable[i])
            grads[i].SetZero();
    }

    for (int i = (int)tape.size() - 1; i >= 0; i--)
    {
        const auto& entry = tape[i];
        const auto& outer_grad = grads[entry.out];

        switch (entry.op)
        {
        case Operator::Add:
            if constexpr (std::is_same_v<T, Tensor>)
            {
                grads[entry.a] += outer_grad;
                grads[entry.b] += outer_grad;
            }
            else
            {
                grads[entry.a] += outer_grad;
                grads[entry.b] += outer_grad;
            }
            break;
        case Operator::Subtract:
            if constexpr (std::is_same_v<T, Tensor>)
            {
                grads[entry.a] += outer_grad;
                grads[entry.b] -= outer_grad;
            }
            else
            {
                grads[entry.a] += outer_grad;
                grads[entry.b] -= outer_grad;
            }
            break;
        case Operator::Multiply:
            if constexpr (std::is_same_v<T, Tensor>)
            {
                Kernels::MatMul_Backward_A(grads[entry.a], outer_grad, values[entry.b]);
                Kernels::MatMul_Backward_B(grads[entry.b], values[entry.a], outer_grad);
            }
            else if constexpr (std::is_same_v<T, Tensor4D>)
            {
                grads[entry.a] += outer_grad % values[entry.b].Transpose();
                grads[entry.b] += values[entry.a].Transpose() % outer_grad;
            }
            else
            {
                grads[entry.a] += outer_grad * values[entry.b].Transpose();
                grads[entry.b] += values[entry.a].Transpose() * outer_grad;
            }
            break;
        case Operator::Sum:
            if constexpr (std::is_same_v<T, Tensor>)
                grads[entry.a] += outer_grad;
            else if constexpr (std::is_same_v<T, Tensor4D>)
                grads[entry.a] += outer_grad;
            else
                grads[entry.a] += outer_grad;
            break;

        case Operator::ElementwiseAdd:
            if constexpr (std::is_same_v<T, Tensor>)
            {
                throw std::runtime_error("Unsupported Operation");
            }
            else if constexpr (std::is_same_v<T, Tensor4D>)
            {
                throw std::runtime_error("Unsupported Operation");
            }
            else
            {
                grads[entry.a] += outer_grad;
                grads[entry.b] += outer_grad.Sum();
            }
            break;
        case Operator::ElementwiseMul:
            if constexpr (std::is_same_v<T, Tensor>)
            {
                grads[entry.a] += values[entry.b].ElementwiseMul(outer_grad);
                grads[entry.b] += values[entry.a].ElementwiseMul(outer_grad);
            }
            else if constexpr (std::is_same_v<T, Tensor4D>)
            {
                grads[entry.a] += values[entry.b] * outer_grad;
                grads[entry.b] += values[entry.a] * outer_grad;
            }
            else
            {
                grads[entry.a] += values[entry.b].ElementwiseMul(outer_grad);
                grads[entry.b] += values[entry.a].ElementwiseMul(outer_grad);
            }
            break;
        case Operator::Pack:
            if constexpr (std::is_same_v<T, Tensor>)
            {
                throw std::runtime_error("Unsupported Operation");
            }
            else if constexpr (std::is_same_v<T, Tensor4D>)
            {
                throw std::runtime_error("Unsupported Operation");
            }
            else
            {
                for (size_t j = 0; j < entry.inputs.size(); j++)
                    grads[entry.inputs[j]] += outer_grad.GetValue()[j];
            }
            break;

        case Operator::Tanh:
            if constexpr (std::is_same_v<T, Tensor>)
                Kernels::Tanh_Backward(grads[entry.a], values[entry.out], outer_grad);
            else if constexpr (std::is_same_v<T, Tensor4D>)
                grads[entry.a] += (1.0f - values[entry.out] * values[entry.out]) * outer_grad;//tanh' = 1 - tanh^2
            else
                grads[entry.a] += (1.0f - values[entry.out].ElementwiseMul(values[entry.out])).ElementwiseMul(outer_grad);//tanh' = 1 - tanh^2
            break;
        case Operator::ReLU:
            if constexpr (std::is_same_v<T, Tensor>)
                grads[entry.a] += values[entry.out].Heaviside().ElementwiseMul(outer_grad);//ReLU' = 1 if x > 0 else 0
            else if constexpr (std::is_same_v<T, Tensor4D>)
                grads[entry.a] += values[entry.out].Heaviside() * outer_grad;//ReLU' = 1 if x > 0 else 0
            else
                grads[entry.a] += values[entry.out].Heaviside().ElementwiseMul(outer_grad);//ReLU' = 1 if x > 0 else 0
            break;

        case Operator::Softmax_CrossEntropy:
        {
            if constexpr (std::is_same_v<T, Tensor>)
            {
                // logits are stored in "left"
                T& logits = values[entry.a];
                T& grad_logits = grads[entry.a];

                const int target = (int)values[entry.b].Data()[0];

                // forward softmax recomputation
                // (oder cached probabilities!)
                float max_val = logits.Max(); // numerical stability

                float sum = 0.0f;
                std::vector<float> probs(logits.GetSize());

                for (size_t i = 0; i < logits.GetSize(); i++)
                {
                    probs[i] = std::exp(logits.Data()[i] - max_val);
                    sum += probs[i];
                }

                for (float& p : probs)
                    p /= sum;

                // backward: dL/dlogits = p - y
                for (size_t i = 0; i < logits.GetSize(); i++)
                {
                    float y = (i == (size_t)target) ? 1.0f : 0.0f;
                    grad_logits.Data()[i] += (probs[i] - y) * outer_grad.Data()[0];
                }
            }
            else if constexpr (std::is_same_v<T, Tensor4D>)
            {
                // logits are stored in "left"
                T& logits = values[entry.a];
                T& grad_logits = grads[entry.a];

                const int target = (int)values[entry.b].Data()[0];

                // forward softmax recomputation
                // (oder cached probabilities!)
                float max_val = logits.Max(); // numerical stability

                float sum = 0.0f;
                std::vector<float> probs(logits.GetSize());

                for (size_t i = 0; i < logits.GetSize(); i++)
                {
                    probs[i] = std::exp(logits.Data()[i] - max_val);
                    sum += probs[i];
                }

                for (float& p : probs)
                    p /= sum;

                // backward: dL/dlogits = p - y
                for (size_t i = 0; i < logits.GetSize(); i++)
                {
                    float y = (i == (size_t)target) ? 1.0f : 0.0f;
                    grad_logits.Data()[i] += (probs[i] - y) * outer_grad.Data()[0];
                }
            }
            else
            {
                // logits are stored in "left"
                T& logits = values[entry.a];
                T& grad_logits = grads[entry.a];

                const int target = (int)values[entry.b].GetValue()[0];

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
                    grad_logits.SetValue()[i] += (probs[i] - y) * outer_grad[0];
                }
            }
            break;
        }
        case Operator::Conv2D:
            if constexpr (std::is_same_v<T, Tensor4D>)
                Kernels::Conv2D_Backward(grads[entry.a], grads[entry.b], values[entry.a], values[entry.b], outer_grad, entry.stride, entry.padding);
            else
                throw std::runtime_error("Unsupported Operation");
            break;
        case Operator::MaxPool2D:
            if constexpr (std::is_same_v<T, Tensor4D>)
                Kernels::MaxPool2D_Backward(grads[entry.a], values[entry.a], outer_grad, values[entry.b], entry.kernel_size, entry.stride);
            else
                throw std::runtime_error("Unsupported Operation");
            break;
        case Operator::Flatten:
            if constexpr (std::is_same_v<T, Tensor4D>)
            {
                grads[entry.a].CopyFrom(grads[entry.out]);
                grads[entry.a] = grads[entry.a].Reshape({ entry.B, entry.C, entry.H, entry.W });
            }
            else
                throw std::runtime_error("Unsupported Operation");
            break;

        default:
            throw std::runtime_error("Unsupported Operation");
        }
    }
}



template<typename T>
inline void TapeRecorder<T>::PrintTape() const
{
    int i = 0;

    for (const auto& entry : tape)
    {
        std::string out_label = "t" + std::to_string(entry.out);
        std::string a_label   = "t" + std::to_string(entry.a);
        std::string b_label   = "t" + std::to_string(entry.b);
        std::string c_label   = "t" + std::to_string(entry.c);

        auto it = id_to_label.find(entry.out);
        if (it != id_to_label.end())
            out_label = it->second;
        it = id_to_label.find(entry.a);
        if (it != id_to_label.end())
            a_label = it->second;
        it = id_to_label.find(entry.b);
        if (it != id_to_label.end())
            b_label = it->second;
        it = id_to_label.find(entry.c);
        if (it != id_to_label.end())
            c_label = it->second;

        std::string op_text = "???";
        std::string op_sign = "";

        switch (entry.op)
        {
        case Operator::Add:
            op_text = "Add";
            op_sign = "+";
            break;
        case Operator::Subtract:
            op_text = "Subtract";
            op_sign = "-";
            break;
        case Operator::Multiply:
            op_text = "Multiply";
            op_sign = "*";
            break;
        case Operator::ElementwiseAdd:
            op_text = "Elementwise-Add";
            op_sign = "+";
            break;
        case Operator::ElementwiseMul:
            op_text = "Elementwise-Multiply";
            op_sign = "*";
            break;
        case Operator::Tanh:
            op_text = "Tanh";
            op_sign = "tanh";
            printf("%.02d: %s: %s [%s] = ", ++i, op_text.c_str(), out_label.c_str(), values[entry.out].Shape2String().c_str());
            printf("tanh(%s [%s])\n", a_label.c_str(), values[entry.a].Shape2String().c_str());
            break;
        case Operator::ReLU:
            op_text = "ReLU";
            op_sign = "relu";
            printf("%.02d: %s: %s [%s] = ", ++i, op_text.c_str(), out_label.c_str(), values[entry.out].Shape2String().c_str());
            printf("ReLU(%s [%s])\n", a_label.c_str(), values[entry.a].Shape2String().c_str());
            break;
        case Operator::Sum:
            op_text = "Sum";
            op_sign = "sum";
            printf("%.02d: %s: %s [%s] = ", ++i, op_text.c_str(), out_label.c_str(), values[entry.out].Shape2String().c_str());
            printf("Sum(%s [%s])\n", a_label.c_str(), values[entry.a].Shape2String().c_str());
            break;
        case Operator::Softmax:
            op_text = "Softmax";
            op_sign = "softmax";
            printf("%.02d: %s: %s [%s] = ", ++i, op_text.c_str(), out_label.c_str(), values[entry.out].Shape2String().c_str());
            printf("softmax(%s [%s], %s [%s])\n", a_label.c_str(), values[entry.a].Shape2String().c_str(), b_label.c_str(), values[entry.b].Shape2String().c_str());
            break;
        case Operator::CrossEntropy:
            op_text = "CrossEntropy";
            op_sign = "crossentropy";
            printf("%.02d: %s: %s [%s] = ", ++i, op_text.c_str(), out_label.c_str(), values[entry.out].Shape2String().c_str());
            printf("CE(%s [%s], %s [%s])\n", a_label.c_str(), values[entry.a].Shape2String().c_str(), b_label.c_str(), values[entry.b].Shape2String().c_str());
            break;
        case Operator::Softmax_CrossEntropy:
            op_text = "Softmax_CrossEntropy";
            op_sign = "smce";
            printf("%.02d: %s: %s [%s] = ", ++i, op_text.c_str(), out_label.c_str(), values[entry.out].Shape2String().c_str());
            printf("SMCE(%s [%s], %s [%s])\n", a_label.c_str(), values[entry.a].Shape2String().c_str(), b_label.c_str(), values[entry.b].Shape2String().c_str());
            break;
        case Operator::Conv2D:
            op_text = "Conv2D";
            op_sign = "conv2d";
            printf("%.02d: %s: %s [%s] = ", ++i, op_text.c_str(), out_label.c_str(), values[entry.out].Shape2String().c_str());
            printf("Conv2D(%s [%s], %s [%s])\n", a_label.c_str(), values[entry.a].Shape2String().c_str(), b_label.c_str(), values[entry.b].Shape2String().c_str());
            break;
        case Operator::MaxPool2D:
            op_text = "MaxPool2D";
            printf("%.02d: %s: %s [%s] = ", ++i, op_text.c_str(), out_label.c_str(), values[entry.out].Shape2String().c_str());
            printf("MaxPool2d(%s [%s]) -> %s [%s]\n", a_label.c_str(), values[entry.a].Shape2String().c_str(), b_label.c_str(), values[entry.b].Shape2String().c_str());
            break;
        case Operator::Flatten:
            op_text = "Flatten";
            op_sign = "flatten";
            printf("%.02d: %s: %s [%s] = ", ++i, op_text.c_str(), out_label.c_str(), values[entry.out].Shape2String().c_str());
            printf("Flatten(%s [%s])\n", a_label.c_str(), values[entry.a].Shape2String().c_str());
            break;
        }


        if (entry.a < 0)
            printf("%.02d: %s: %s [%s] %s\n", ++i, op_text.c_str(),
                out_label.c_str(), values[entry.out].Shape2String().c_str(),
                op_sign.c_str());

        else if (entry.b < 0)
            printf("%.02d: %s: %s [%s] = %s(%s [%s])\n", ++i, op_text.c_str(),
                out_label.c_str(), values[entry.out].Shape2String().c_str(),
                op_sign.c_str(),
                a_label.c_str(), values[entry.a].Shape2String().c_str());
        
        else if (entry.c < 0)
            printf("%.02d: %s: %s [%s] = %s [%s] %s %s [%s]\n", ++i, op_text.c_str(),
                out_label.c_str(), values[entry.out].Shape2String().c_str(),
                a_label.c_str(), values[entry.a].Shape2String().c_str(),
                op_sign.c_str(),
                b_label.c_str(), values[entry.b].Shape2String().c_str());

        else
            printf("%.02d: %s: %s [%s] = %s [%s] %s %s [%s] -> %s [%s]\n", ++i, op_text.c_str(),
                out_label.c_str(), values[entry.out].Shape2String().c_str(),
                a_label.c_str(), values[entry.a].Shape2String().c_str(),
                op_sign.c_str(),
                b_label.c_str(), values[entry.b].Shape2String().c_str(),
                c_label.c_str(), values[entry.c].Shape2String().c_str());
    }
}