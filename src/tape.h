#pragma once
#include <unordered_map>
#include <string>
#include <stdio.h>
#include <fstream>
#include "node.h"
#include "tensor4d.h"
#include "kernels.h"
#include "thread_pool.h"



struct TapeEntry
{
    Operator op;

    int a = -1;   // input tensor index
    int b = -1;   // optional
    int c = -1;   // optional, used for cache
    std::vector<int> inputs;

    int out; // output tensor index

    // Convolution
    int stride  = 1;
    int padding = 0;

	// Max Pool
	int kernel_size = 1;

    // Flatten
    size_t B = 0;
    size_t C = 0;
    size_t H = 0;
    size_t W = 0;


    void Save(std::ostream& os) const
    {
        os.write((char*)&op, sizeof(op));

        os.write((char*)&a, sizeof(a));
        os.write((char*)&b, sizeof(b));
        os.write((char*)&c, sizeof(c));
        os.write((char*)&out, sizeof(out));

        os.write((char*)&stride, sizeof(stride));
        os.write((char*)&padding, sizeof(padding));
        os.write((char*)&kernel_size, sizeof(kernel_size));

        os.write((char*)&B, sizeof(B));
        os.write((char*)&C, sizeof(C));
        os.write((char*)&H, sizeof(H));
        os.write((char*)&W, sizeof(W));

        size_t n = inputs.size();
        os.write((char*)&n, sizeof(n));

        if (n)
            os.write((char*)inputs.data(), sizeof(int) * n);
    }


    void Load(std::istream& is)
    {
        is.read((char*)&op, sizeof(op));

        is.read((char*)&a, sizeof(a));
        is.read((char*)&b, sizeof(b));
        is.read((char*)&c, sizeof(c));
        is.read((char*)&out, sizeof(out));

        is.read((char*)&stride, sizeof(stride));
        is.read((char*)&padding, sizeof(padding));
        is.read((char*)&kernel_size, sizeof(kernel_size));

        is.read((char*)&B, sizeof(B));
        is.read((char*)&C, sizeof(C));
        is.read((char*)&H, sizeof(H));
        is.read((char*)&W, sizeof(W));

        size_t n;
        is.read((char*)&n, sizeof(n));

        inputs.resize(n);

        if (n)
            is.read((char*)inputs.data(), sizeof(int) * n);
    }
};



struct Metadata
{
    bool trainable = false;
    bool requires_grad = false;
    bool has_weight_decay = false;//Is this parameter included in regularization?

    bool is_cache = false;
    bool calibrated = false;//For BatchNorm, true if calibration has been done

    std::string label = "";
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

    void StartCalibration();
    void EndCalibration();

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
        return metadata[Index].trainable;
    }

    bool RequiresGradient(size_t Index) const
    {
        return metadata[Index].requires_grad;
    }

    bool HasWeightDecay(size_t Index) const
    {
        return metadata[Index].has_weight_decay;
    }

    size_t GetNumberOfValues() const
    {
        return values.size();
    }

    int AddDataEntry(const T& v, bool Trainable = false, bool WeightDecay = false)
    {
        values.push_back(v);
        grads.push_back(v.Clone());

        Metadata new_metadata;
        new_metadata.trainable     = Trainable;
        new_metadata.requires_grad = false;
        new_metadata.has_weight_decay  = false;
        metadata.emplace_back(new_metadata);

        return (int)values.size() - 1;
    }

    void AddOpEntry(Operator op, int a, int out)
    {
        tape.push_back({ op, a, -1, -1, {}, out });
    }

	void AddOpEntry(Operator op, int a, int b, int out)
	{
        tape.push_back({ op, a, b, -1, {}, out });
	}

    void AddOpEntry(Operator op, int a, int b, int c, int out)
    {
        tape.push_back({ op, a, b, c, {}, out });
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
    void PrintArchitecture() const;

    void SaveToFile(const std::string& filename) const;
    bool LoadFromFile(const std::string& filename);

private:
    void Visit(const NodePtr<T>& node, std::unordered_map<NodePtr<T>, int>& node_to_id);

	std::vector<TapeEntry> tape;

    std::vector<T> values;
    std::vector<T> grads;
    std::vector<T> calibrations_mean;
    std::vector<T> calibrations_var;
    std::vector<Metadata> metadata;

    bool m_CalibrationStarted = false;
    bool m_IsCalibrated = false;
    size_t m_CalibrationBatchesSeen = 0;

    ThreadPool thread_pool = ThreadPool(8);

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


    auto node_id = AddDataEntry(node->GetValue(), node->IsTrainable(), node->HasWeightDecay());
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

	// One operand
	case Operator::Sum:
    case Operator::Sigmoid:
    case Operator::Tanh:
    case Operator::ReLU:
    case Operator::Softmax:
    case Operator::GlobalAveragePool2D:
        AddOpEntry(node->op, GetID(node->left), node_id);
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

    // Batch norms
    case Operator::BatchNorm:
    {
        if constexpr (std::is_same_v<T, Tensor4D>)
        {
            const auto& value = node->GetValue();
            auto cache_id_mean = AddDataEntry(Tensor4D({ 1, value.GetDepth(), value.GetRows(), value.GetColumns() }));
            auto cache_id_std = AddDataEntry(Tensor4D({ 1, value.GetDepth(), value.GetRows(), value.GetColumns() }));
            AddOpEntry(node->op, GetID(node->left), cache_id_mean, cache_id_std, node_id);
        }
        else
            throw std::runtime_error("Unsupported Operation");
        break;
    }
    case Operator::BatchNorm2D:
    {
        if constexpr (std::is_same_v<T, Tensor4D>)
        {
            const auto& value  = node->GetValue();
            auto cache_id_mean = AddDataEntry(Tensor4D({ 1, value.GetDepth(), 1, 1 }));
            auto cache_id_std  = AddDataEntry(Tensor4D({ 1, value.GetDepth(), 1, 1 }));
            AddOpEntry(node->op, GetID(node->left), cache_id_mean, cache_id_std, node_id);
        }
        else
            throw std::runtime_error("Unsupported Operation");
        break;
    }

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

    // Mark trainable tensors as requiring gradients
    for (size_t i = 0; i < grads.size() - 1; i++)
        if (metadata[i].trainable)
            metadata[i].requires_grad = true;

    // Calculate all tensors that require gradients

    bool any_change;

    do
    {
        any_change = false;

        for (size_t i = 0; i < tape.size(); i++)
        {
            if ( (tape[i].a >= 0 && metadata[tape[i].a].requires_grad) || (tape[i].b >= 0 && metadata[tape[i].b].requires_grad) )
            {
                if (!metadata[tape[i].out].requires_grad)
                {
                    metadata[tape[i].out].requires_grad = true;
                    any_change = true;
                }
            }
        }
    } while (any_change);
}



template<typename T>
void TapeRecorder<T>::Forward()
{
    int calibration_index = 0;

	for (const auto& entry : tape)
	{
        switch (entry.op)
        {
		case Operator::Add:
            values[entry.out] = values[entry.a] + values[entry.b];
			break;
		case Operator::Subtract:
            if constexpr (std::is_same_v<T, Tensor4D>)
            {
                //values[entry.a].Print();
                //values[entry.b].Print();
                values[entry.out] = values[entry.a] - values[entry.b];
                //values[entry.out].Print();
            }
            else
                values[entry.out] = values[entry.a] - values[entry.b];
            break;
        case Operator::Multiply:
            if constexpr (std::is_same_v<T, Tensor>)
                Kernels::MatMul_Forward(values[entry.out], values[entry.a], values[entry.b]);
            else if constexpr (std::is_same_v<T, Tensor4D>)
            {
                //values[entry.a].Print();
                //values[entry.b].Print();
                //values[entry.out] = values[entry.a] % values[entry.b];
                Kernels::MatMul_Forward(values[entry.out], values[entry.a], values[entry.b]);
                //values[entry.out].Print();
            }
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
                //values[entry.a].Print();
                //values[entry.b].Print();
                values[entry.out] = values[entry.a] * values[entry.b];
                //values[entry.out].Print();
            }
            else
                values[entry.out] = values[entry.a].ElementwiseMul(values[entry.b]);
            break;

        case Operator::Pack:
            if constexpr (std::is_same_v<T, Tensor>)
                throw std::runtime_error("Unsupported Operation");
            else if constexpr (std::is_same_v<T, Tensor4D>)
                throw std::runtime_error("Unsupported Operation");
            else
            {
                values[entry.out].SetLength(entry.inputs.size());//Neccessary???
                for (size_t j = 0; j < entry.inputs.size(); j++)
                    values[entry.out].SetValue()[j] = values[entry.inputs[j]].GetValue()[0];
            }
            break;

        case Operator::Sigmoid:
            if constexpr (std::is_same_v<T, Tensor4D>)
                values[entry.out] = values[entry.a].Sigmoid();
            else
                throw std::runtime_error("Unsupported Operation");
            break;
        case Operator::Tanh:
            values[entry.out] = values[entry.a].Tanh();
            break;
        case Operator::ReLU:
            values[entry.out] = values[entry.a].ReLU();
            break;
        case Operator::Softmax_CrossEntropy:
            if constexpr (std::is_same_v<T, Tensor>)
                values[entry.out] = values[entry.a].Softmax().CrossEntropy(values[entry.b]);
            else if constexpr (std::is_same_v<T, Tensor4D>)
                values[entry.out] = values[entry.a].Softmax_CrossEntropy(values[entry.b]);
            else
                values[entry.out] = values[entry.a].Softmax().CrossEntropy(values[entry.b]);
            break;
        case Operator::Conv2D:
            if constexpr (std::is_same_v<T, Tensor4D>)
            {
                Kernels::Conv2D_Forward(thread_pool, values[entry.out], values[entry.a], values[entry.b], entry.stride, entry.padding);
                //Kernels::Conv2D_Forward(values[entry.out], values[entry.a], values[entry.b], entry.stride, entry.padding);
                //values[entry.a].Print();
                //values[entry.b].Print();
                //values[entry.out].Print();
            }
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
                values[entry.out] = values[entry.a].Reshape({ entry.B, 1, entry.C * entry.H * entry.W, 1 });
            else
                throw std::runtime_error("Unsupported Operation");
            break;
        case Operator::BatchNorm:
            if constexpr (std::is_same_v<T, Tensor4D>)
            {
                if (m_IsCalibrated)
                    values[entry.out] = values[entry.a].BatchNorm(values[entry.b], values[entry.c]);
                else
                {
                    values[entry.out] = values[entry.a].BatchNorm(&values[entry.b], &values[entry.c]);
                    if (m_CalibrationStarted)
                    {
                        calibrations_mean[calibration_index] += values[entry.b];
                        calibrations_var[calibration_index] += values[entry.c] * values[entry.c];
                        calibration_index++;
                    }
                }
            }
            else
                throw std::runtime_error("Unsupported Operation");
            break;
        case Operator::BatchNorm2D:
            if constexpr (std::is_same_v<T, Tensor4D>)
            {
                if (m_IsCalibrated)
                    values[entry.out] = values[entry.a].BatchNorm2D(values[entry.b], values[entry.c]);
                else
                {
                    values[entry.out] = values[entry.a].BatchNorm2D(&values[entry.b], &values[entry.c]);
                    if (m_CalibrationStarted)
                    {
                        calibrations_mean[calibration_index] += values[entry.b];
                        calibrations_var[calibration_index] += values[entry.c] * values[entry.c];
                        calibration_index++;
                    }
                }
            }
            else
                throw std::runtime_error("Unsupported Operation");
            break;
        case Operator::GlobalAveragePool2D:
            if constexpr (std::is_same_v<T, Tensor4D>)
            {
                values[entry.out] = values[entry.a].GlobalAveragePool2D();
            }
            else
                throw std::runtime_error("Unsupported Operation");
            break;

        default:
            throw std::runtime_error("Unsupported Operation");
        }
	}

    if (m_CalibrationStarted)
        m_CalibrationBatchesSeen++;
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
        if (!metadata[i].trainable)
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
                grads[entry.a] += outer_grad % values[entry.b].Transpose();//Move to kernel
                grads[entry.b] += values[entry.a].Transpose() % outer_grad;//Move to kernel
                //Kernels::MatMul_Backward_A(grads[entry.a], outer_grad, values[entry.b]);//BUG
                //Kernels::MatMul_Backward_B(grads[entry.b], values[entry.a], outer_grad);//BUG
                //values[entry.b].Print();
                //grads[entry.a].Print();
                //grads[entry.b].Print();
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

        case Operator::Sigmoid:
            if constexpr (std::is_same_v<T, Tensor4D>)
                grads[entry.a] += values[entry.out] * (1.0f - values[entry.out]) * outer_grad;//sigmoid' = sigmoid (1 - sigmoid)
            else
                throw std::runtime_error("Unsupported Operation");
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

                const int target = (int)values[entry.b].Data()[0];// Works only with batch size 1

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
                Kernels::Softmax_CrossEntropy_Backward(values[entry.a], grads[entry.a], values[entry.b], outer_grad);
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
            {
                //outer_grad.Print();
                //Kernels::Conv2D_Backward(grads[entry.a], grads[entry.b], values[entry.a], values[entry.b], outer_grad, entry.stride, entry.padding);
                Kernels::Conv2D_Backward(thread_pool, grads[entry.a], grads[entry.b], values[entry.a], values[entry.b], outer_grad, entry.stride, entry.padding);
            }
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
                grads[entry.a] += grads[entry.out].Reshape({ entry.B, entry.C, entry.H, entry.W });
            }
            else
                throw std::runtime_error("Unsupported Operation");
            break;
        case Operator::BatchNorm:
            if constexpr (std::is_same_v<T, Tensor4D>)
            {
                Kernels::BatchNorm_Backward(grads[entry.a], values[entry.a], outer_grad, values[entry.b], values[entry.c]);
            }
            else
                throw std::runtime_error("Unsupported Operation");
            break;
        case Operator::BatchNorm2D:
            if constexpr (std::is_same_v<T, Tensor4D>)
            {
                Kernels::BatchNorm2D_Backward(grads[entry.a], values[entry.a], outer_grad, values[entry.b], values[entry.c]);
            }
            else
                throw std::runtime_error("Unsupported Operation");
            break;
        case Operator::GlobalAveragePool2D:
            if constexpr (std::is_same_v<T, Tensor4D>)
            {
                float inv = 1.0f / ((float)values[entry.out].GetRows() * values[entry.out].GetColumns());
                grads[entry.a] += inv * grads[entry.out];
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
inline void TapeRecorder<T>::StartCalibration()
{
    calibrations_mean.clear();
    calibrations_var.clear();

    for (int i = 0; i < (int)tape.size(); i++)
    {
        const auto& entry = tape[i];

        if (entry.op == Operator::BatchNorm)
        {
            auto shape = values[entry.a].GetShape();
            calibrations_mean.push_back(Tensor4D({ 1, shape[1], shape[2], shape[3] }));
            calibrations_var .push_back(Tensor4D({ 1, shape[1], shape[2], shape[3] }));
        }

        else if (entry.op == Operator::BatchNorm2D)
        {
            auto shape = values[entry.a].GetShape();
            calibrations_mean.push_back(Tensor4D({ 1, shape[1], 1, 1 }));
            calibrations_var .push_back(Tensor4D({ 1, shape[1], 1, 1 }));
        }
    }

    m_CalibrationStarted = true;
    m_CalibrationBatchesSeen = 0;
}



template<typename T>
inline void TapeRecorder<T>::EndCalibration()
{
    if (m_CalibrationBatchesSeen == 0)
        throw std::runtime_error("No calibration batches seen.");

    float inv_batch_count = 1.0f / (float)m_CalibrationBatchesSeen;

    for (auto& tensor : calibrations_mean)
        tensor = inv_batch_count * tensor;

    for (auto& tensor : calibrations_var)
        tensor = inv_batch_count * tensor;

    int j = 0;

    for (int i = 0; i < (int)tape.size(); i++)
    {
        const auto& entry = tape[i];

        if (entry.op == Operator::BatchNorm)
        {
            values[entry.b].CopyFrom(calibrations_mean[j]);
            values[entry.c].CopyFrom(calibrations_var[j].Sqrt());
            j++;
        }

        else if (entry.op == Operator::BatchNorm2D)
        {
            values[entry.b].CopyFrom(calibrations_mean[j]);
            values[entry.c].CopyFrom(calibrations_var[j].Sqrt());
            j++;
        }
    }
    

    m_CalibrationStarted = false;
    m_IsCalibrated = true;
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
            printf("%.02d: %s: %s [%s] = ", ++i, op_text.c_str(), out_label.c_str(), values[entry.out].Shape2String().c_str());
            printf("%s [%s] + %s [%s]\n", a_label.c_str(), values[entry.a].Shape2String().c_str(), b_label.c_str(), values[entry.b].Shape2String().c_str());
            break;
        case Operator::Subtract:
            op_text = "Subtract";
            printf("%.02d: %s: %s [%s] = ", ++i, op_text.c_str(), out_label.c_str(), values[entry.out].Shape2String().c_str());
            printf("%s [%s] - %s [%s]\n", a_label.c_str(), values[entry.a].Shape2String().c_str(), b_label.c_str(), values[entry.b].Shape2String().c_str());
            break;
        case Operator::Multiply:
            op_text = "Multiply";
            printf("%.02d: %s: %s [%s] = ", ++i, op_text.c_str(), out_label.c_str(), values[entry.out].Shape2String().c_str());
            printf("%s [%s] %% %s [%s]\n", a_label.c_str(), values[entry.a].Shape2String().c_str(), b_label.c_str(), values[entry.b].Shape2String().c_str());
            break;
        case Operator::ElementwiseAdd:
            op_text = "Elementwise-Add";
            printf("%.02d: %s: %s [%s] = ", ++i, op_text.c_str(), out_label.c_str(), values[entry.out].Shape2String().c_str());
            printf("%s [%s] + %s [%s]\n", a_label.c_str(), values[entry.a].Shape2String().c_str(), b_label.c_str(), values[entry.b].Shape2String().c_str());
            break;
        case Operator::ElementwiseMul:
            op_text = "Elementwise-Multiply";
            printf("%.02d: %s: %s [%s] = ", ++i, op_text.c_str(), out_label.c_str(), values[entry.out].Shape2String().c_str());
            printf("%s [%s] * %s [%s]\n", a_label.c_str(), values[entry.a].Shape2String().c_str(), b_label.c_str(), values[entry.b].Shape2String().c_str());
            break;
        case Operator::Tanh:
            op_text = "Tanh";
            printf("%.02d: %s: %s [%s] = ", ++i, op_text.c_str(), out_label.c_str(), values[entry.out].Shape2String().c_str());
            printf("tanh(%s [%s])\n", a_label.c_str(), values[entry.a].Shape2String().c_str());
            break;
        case Operator::ReLU:
            op_text = "ReLU";
            printf("%.02d: %s: %s [%s] = ", ++i, op_text.c_str(), out_label.c_str(), values[entry.out].Shape2String().c_str());
            printf("ReLU(%s [%s])\n", a_label.c_str(), values[entry.a].Shape2String().c_str());
            break;
        case Operator::Sum:
            op_text = "Sum";
            printf("%.02d: %s: %s [%s] = ", ++i, op_text.c_str(), out_label.c_str(), values[entry.out].Shape2String().c_str());
            printf("Sum(%s [%s])\n", a_label.c_str(), values[entry.a].Shape2String().c_str());
            break;
        case Operator::Softmax:
            op_text = "Softmax";
            printf("%.02d: %s: %s [%s] = ", ++i, op_text.c_str(), out_label.c_str(), values[entry.out].Shape2String().c_str());
            printf("softmax(%s [%s], %s [%s])\n", a_label.c_str(), values[entry.a].Shape2String().c_str(), b_label.c_str(), values[entry.b].Shape2String().c_str());
            break;
        case Operator::CrossEntropy:
            op_text = "CrossEntropy";
            printf("%.02d: %s: %s [%s] = ", ++i, op_text.c_str(), out_label.c_str(), values[entry.out].Shape2String().c_str());
            printf("CE(%s [%s], %s [%s])\n", a_label.c_str(), values[entry.a].Shape2String().c_str(), b_label.c_str(), values[entry.b].Shape2String().c_str());
            break;
        case Operator::Softmax_CrossEntropy:
            op_text = "Softmax_CrossEntropy";
            printf("%.02d: %s: %s [%s] = ", ++i, op_text.c_str(), out_label.c_str(), values[entry.out].Shape2String().c_str());
            printf("SMCE(%s [%s], %s [%s])\n", a_label.c_str(), values[entry.a].Shape2String().c_str(), b_label.c_str(), values[entry.b].Shape2String().c_str());
            break;
        case Operator::Conv2D:
            op_text = "Conv2D";
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
            printf("%.02d: %s: %s [%s] = ", ++i, op_text.c_str(), out_label.c_str(), values[entry.out].Shape2String().c_str());
            printf("Flatten(%s [%s])\n", a_label.c_str(), values[entry.a].Shape2String().c_str());
            break;
        case Operator::BatchNorm:
            op_text = "BatchNorm";
            printf("%.02d: %s: %s [%s] = ", ++i, op_text.c_str(), out_label.c_str(), values[entry.out].Shape2String().c_str());
            printf("BatchNorm(%s [%s]) -> %s [%s], %s [%s]\n", a_label.c_str(), values[entry.a].Shape2String().c_str(), b_label.c_str(), values[entry.b].Shape2String().c_str(), c_label.c_str(), values[entry.c].Shape2String().c_str());
            break;
        case Operator::BatchNorm2D:
            op_text = "BatchNorm2D";
            printf("%.02d: %s: %s [%s] = ", ++i, op_text.c_str(), out_label.c_str(), values[entry.out].Shape2String().c_str());
            printf("BatchNorm2D(%s [%s]) -> %s [%s], %s [%s]\n", a_label.c_str(), values[entry.a].Shape2String().c_str(), b_label.c_str(), values[entry.b].Shape2String().c_str(), c_label.c_str(), values[entry.c].Shape2String().c_str());
            break;
        case Operator::GlobalAveragePool2D:
            op_text = "GlobalAveragePool2D";
            printf("%.02d: %s: %s [%s] = ", ++i, op_text.c_str(), out_label.c_str(), values[entry.out].Shape2String().c_str());
            printf("GlobalAveragePool2D(%s [%s])\n", a_label.c_str(), values[entry.a].Shape2String().c_str());
            break;
        default:
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

    printf("\n---Data---\n\n");


    for (size_t i = 0; i < values.size(); i++)
    {
        std::string label = "t" + std::to_string(i);

        auto it = id_to_label.find((int)i);
        if (it != id_to_label.end())
            label = it->second;

        std::string trainable = "Trainable";
        std::string requires_gradient = "Requires_Gradient";
        std::string weight_decay = "Weight_Decay";

        if (!IsTrainable(i))
            trainable = "";
        if (!RequiresGradient(i))
            requires_gradient = "";
        if (!HasWeightDecay(i))
            weight_decay = "";

        printf("%.02d: %s %s: %s %s %s\n", (int)i, label.c_str(), values[i].Shape2String().c_str(), trainable.c_str(), requires_gradient.c_str(), weight_decay.c_str());
    }

    printf("\n");
}



template<typename T>
inline void TapeRecorder<T>::PrintArchitecture() const
{
    int i = 0;

    for (const auto& entry : tape)
    {     
        switch (entry.op)
        {
        case Operator::Add:
        case Operator::Subtract:
        case Operator::ElementwiseAdd:
        case Operator::ElementwiseMul:
        case Operator::Sum:
            break;

        case Operator::Multiply:
            if (i > 0)
                printf("-> ");
            printf("Dense");
            i++;
            break;
        case Operator::Tanh:
            if (i > 0)
                printf("-> ");
            printf("Tanh");
            i++;
            break;
        case Operator::ReLU:
            if (i > 0)
                printf("-> ");
            printf("ReLU");
            i++;
            break;
        case Operator::Softmax:
            if (i > 0)
                printf("-> ");
            printf("Softmax");
            i++;
            break;
        case Operator::CrossEntropy:
            if (i > 0)
                printf("-> ");
            printf("CrossEntropy");
            i++;
            break;
        case Operator::Softmax_CrossEntropy:
            if (i > 0)
                printf("-> ");
            printf("Softmax_CrossEntropy");
            i++;
            break;
        case Operator::Conv2D:
            if (i > 0)
                printf("-> ");
            printf("Conv2D");
            i++;
            break;
        case Operator::MaxPool2D:
            if (i > 0)
                printf("-> ");
            printf("MaxPool2D");
            i++;
            break;
        case Operator::Flatten:
            if (i > 0)
                printf("-> ");
            printf("Flatten");
            i++;
            break;
        case Operator::BatchNorm:
        case Operator::BatchNorm2D:
            if (i > 0)
                printf("-> ");
            printf("BatchNorm");
            i++;
            break;            
        }
    }

    printf("\n");
}



template<typename T>
inline void TapeRecorder<T>::SaveToFile(const std::string& filename) const
{
    std::ofstream file(filename, std::ios::binary);

    if (!file)
        throw std::runtime_error("Cannot open file");

    uint32_t magic = 0x54415045; // "TAPE"

    file.write((char*)&magic, sizeof(magic));

    size_t tape_size = tape.size();
    file.write((char*)&tape_size, sizeof(tape_size));

    for (auto& e : tape)
        e.Save(file);

    size_t value_count = values.size();
    file.write((char*)&value_count, sizeof(value_count));

    for (size_t i = 0; i < values.size(); i++)
    {
        uint8_t t = metadata[i].trainable ? 1 : 0;
        file.write((char*)(&t), sizeof(uint8_t));
        t = metadata[i].requires_grad ? 1 : 0;
        file.write((char*)(&t), sizeof(uint8_t));
        t = metadata[i].has_weight_decay ? 1 : 0;
        file.write((char*)(&t), sizeof(uint8_t));

        values[i].GetShape().Save(file);

        if (metadata[i].trainable)
            values[i].Save(file);
    }

    size_t label_to_id_count = label_to_id.size();
    file.write((char*)&label_to_id_count, sizeof(label_to_id_count));

    for (auto& [label, id] : label_to_id)
    {
        file.write((char*)&id, sizeof(id));

        size_t len = label.size();
        file.write((char*)&len, sizeof(len));
        file.write(label.data(), len);
    }


    size_t id_count = id_to_label.size();
    file.write((char*)&id_count, sizeof(id_count));

    for (auto& [id, label] : id_to_label)
    {
        file.write((char*)&id, sizeof(id));

        size_t len = label.size();
        file.write((char*)&len, sizeof(len));
        file.write(label.data(), len);
    }
}



template<typename T>
inline bool TapeRecorder<T>::LoadFromFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file)
        return false;

    uint32_t magic;
    file.read((char*)&magic, sizeof(magic));

    if (magic != 0x54415045)
        return false;

    size_t tape_size;
    file.read((char*)&tape_size, sizeof(tape_size));

    tape.resize(tape_size);

    for (auto& e : tape)
        e.Load(file);

    size_t value_count;
    file.read((char*)&value_count, sizeof(value_count));

    values.resize(value_count);
    grads.resize(value_count);
    metadata.resize(value_count);

    for (size_t i = 0; i < value_count; i++)
    {
        uint8_t tmp;
        file.read((char*)&tmp, sizeof(uint8_t));
        metadata[i].trainable = tmp == 1;
        file.read((char*)&tmp, sizeof(uint8_t));
        metadata[i].requires_grad = tmp == 1;
        file.read((char*)&tmp, sizeof(uint8_t));
        metadata[i].has_weight_decay = tmp == 1;
        

        Tensor4D::Shape shape(file);

        if (metadata[i].trainable)
            values[i] = Tensor4D(shape, file);
        else
            values[i] = Tensor4D(shape);

        grads[i] = values[i].Clone();
    }


    size_t label_to_id_count;
    file.read((char*)&label_to_id_count, sizeof(label_to_id_count));

    label_to_id.clear();

    for (size_t i = 0;i < label_to_id_count;i++)
    {
        int id;
        file.read((char*)&id, sizeof(id));

        size_t len;
        file.read((char*)&len, sizeof(len));

        std::string label(len, '\0');
        file.read(label.data(), len);

        label_to_id[label] = id;
    }


    size_t id_count;
    file.read((char*)&id_count, sizeof(id_count));

    id_to_label.clear();

    for (size_t i = 0;i < id_count;i++)
    {
        int id;
        file.read((char*)&id, sizeof(id));

        size_t len;
        file.read((char*)&len, sizeof(len));

        std::string label(len, '\0');
        file.read(label.data(), len);

        id_to_label[id] = std::move(label);
    }

    return true;
}