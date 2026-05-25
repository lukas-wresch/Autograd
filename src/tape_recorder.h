#pragma once
#include <unordered_map>
#include <string>
#include <stdio.h>
#include "node.h"



struct TapeEntry
{
    Operator op;

    int a;   // input tensor index
    int b;   // optional
    std::vector<int> inputs;

    int out; // output tensor index
};



template<typename T>
class TapeRecorder
{
public:
    TapeRecorder() = default;
    TapeRecorder(const NodePtr<T>& root) { Compile(root); }

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

        T new_grad(v);//initialize gradient with same length
        new_grad.SetZero();
		grads.push_back(new_grad);

        trainable.push_back(Trainable);

        return (int)values.size() - 1;
    }

	void AddOpEntry(Operator op, int a, int b, int out)
	{
        tape.push_back({ op, a, b, {}, out });
	}

    void AddOpEntry(Operator op, const std::vector<int>& inputs, int out)
    {
        tape.push_back({ op, -1, -1, inputs, out });
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

    //Two operands
    case Operator::Add:
    case Operator::Subtract:
    case Operator::Multiply:
    case Operator::ElementwiseAdd:
    case Operator::ElementwiseMul:
        AddOpEntry(node->op, GetID(node->left), GetID(node->right), node_id);
        break;

	//1 Operand
	case Operator::Sum:
    case Operator::Tanh:
    case Operator::ReLU:
        AddOpEntry(node->op, GetID(node->left), -1, node_id);
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
            values[entry.out] = values[entry.a] * values[entry.b];
            break;

        case Operator::Sum:
            values[entry.out] = values[entry.a].Sum();
            break;
        case Operator::ElementwiseAdd:
            values[entry.out] = values[entry.a].ElementwiseAdd(values[entry.b]);
            break;
        case Operator::ElementwiseMul:
            values[entry.out] = values[entry.a].ElementwiseMul(values[entry.b]);
            break;

        case Operator::Pack:
            values[entry.out].SetLength(entry.inputs.size());//Neccessary???
            for (size_t j = 0; j < entry.inputs.size(); j++)
                values[entry.out].SetValue()[j] = values[entry.inputs[j]].GetValue()[0];
            break;

        case Operator::Tanh:
            values[entry.out] = values[entry.a].Tanh();
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
            grads[entry.a] += outer_grad;
            grads[entry.b] += outer_grad;
            break;
        case Operator::Subtract:
            grads[entry.a] += outer_grad;
            grads[entry.b] -= outer_grad;
            break;
        case Operator::Multiply:
            grads[entry.a] += outer_grad * values[entry.b].Transpose();
            grads[entry.b] += values[entry.a].Transpose() * outer_grad;
            break;
        case Operator::Sum:
            //grads[entry.a] = grads[entry.a].ElementwiseAdd(outer_grad);
            grads[entry.a] += outer_grad;
            break;

        case Operator::ElementwiseAdd:
            grads[entry.a] += outer_grad;
            grads[entry.b] += outer_grad.Sum();
            break;
        case Operator::ElementwiseMul:
            grads[entry.a] += values[entry.b].ElementwiseMul(outer_grad);
            grads[entry.b] += values[entry.a].ElementwiseMul(outer_grad);
            break;
        case Operator::Pack:
            for (size_t j = 0; j < entry.inputs.size(); j++)
                grads[entry.inputs[j]] += outer_grad.GetValue()[j];
            break;

        case Operator::Tanh:
            grads[entry.a] += (1.0f - values[entry.out].ElementwiseMul(values[entry.out])).ElementwiseMul(outer_grad);//tanh' = 1 - tanh^2
            break;
        case Operator::ReLU:
            grads[entry.a] += values[entry.out].Heaviside().ElementwiseMul(outer_grad);//ReLU' = 1 if x > 0 else 0
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
        case Operator::Tanh:
            op_text = "tanh";
            op_sign = "tanh";
            break;
        }

        std::string out_label = std::to_string(entry.out);
        std::string a_label = std::to_string(entry.a);
        std::string b_label = std::to_string(entry.b);

        auto it = id_to_label.find(entry.out);
        if (it != id_to_label.end())
            out_label = it->second;
        it = id_to_label.find(entry.a);
        if (it != id_to_label.end())
            a_label = it->second;
        it = id_to_label.find(entry.b);
        if (it != id_to_label.end())
            b_label = it->second;


        if (entry.a < 0)
            printf("%.02d: %s %s [%dx%d] %s\n", ++i, op_text.c_str(),
                out_label.c_str(), (int)values[entry.out].GetRows(), (int)values[entry.out].GetColumns(),
                op_sign.c_str());

        else if (entry.b < 0)
            printf("%.02d: %s %s [%dx%d] = %s [%dx%d] %s\n", ++i, op_text.c_str(),
                out_label.c_str(), (int)values[entry.out].GetRows(), (int)values[entry.out].GetColumns(),
                a_label.c_str(), (int)values[entry.a].GetRows(), (int)values[entry.a].GetColumns(),
                op_sign.c_str());
        
        else
            printf("%.02d: %s %s [%dx%d] = %s [%dx%d] %s %s [%dx%d]\n", ++i, op_text.c_str(),
                out_label.c_str(), (int)values[entry.out].GetRows(), (int)values[entry.out].GetColumns(),
                a_label.c_str(), (int)values[entry.a].GetRows(), (int)values[entry.a].GetColumns(),
                op_sign.c_str(),
                b_label.c_str(), (int)values[entry.b].GetRows(), (int)values[entry.b].GetColumns());
    }
}