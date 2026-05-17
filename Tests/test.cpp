#include "gtest/gtest.h"
#include <random>
#include "../node.h"
#include "../neuron.h"
#include "../layer.h"
#include "../sgd.h"



void ShuffleDataset(std::vector<NodePtr<VectorType>>& xs, std::vector<NodePtr<VectorType>>& labels)
{
	static std::random_device rd;
	static std::mt19937 rng(rd());

	for (size_t i = xs.size() - 1; i > 0; i--)
	{
		std::uniform_int_distribution<size_t> dist(0, i);
		size_t j = dist(rng);

		std::swap(xs[i], xs[j]);
		std::swap(labels[i], labels[j]);
	}
}



TEST(ScalarTests, BasicArithmeticAndBackprop)
{ // d = a * b + c
	auto a = Node<ScalarType>::Create(2.0f);
	auto b = Node<ScalarType>::Create(-3.0f);
	auto c = Node<ScalarType>::Create(10.0f);

	auto d = a * b + c;
	d->Backwards();

	EXPECT_NEAR(d->GetValue().GetValue()[0], 2.0f * -3.0f + 10.0f, 1e-6f);
	EXPECT_NEAR(d->GetGradient().GetValue()[0], 1.0f, 1e-6f); // dd/dd = 1
	EXPECT_NEAR(a->GetGradient().GetValue()[0], -3.0f, 1e-6f); // dd/da = b
	EXPECT_NEAR(b->GetGradient().GetValue()[0], 2.0f, 1e-6f);  // dd/db = a
	EXPECT_NEAR(c->GetGradient().GetValue()[0], 1.0f, 1e-6f);  // dd/dc = 1
}



TEST(VectorTests, ElementwiseMultiplyAddBackprop)
{ // out = (w * x) + b2 (elementwise add with scalar bias)
	auto x  = Node<VectorType>::Create({ 0.5f, 0.0f });
	auto w  = Node<VectorType>::Create({ 1.0f, 0.5f });
	auto b2 = Node<VectorType>::Create({ 1.0f });

	auto out = (w * x)->ElementwiseAdd(b2);

	// forward value checks
	EXPECT_NEAR(out->GetValue().m_pValues[0], 1.0f * 0.5f + 1.0f, 1e-6f);
	EXPECT_NEAR(out->GetValue().m_pValues[1], 0.5f * 0.0f + 1.0f, 1e-6f);

	out->Backwards();

	// After Backwards:
	// - gradient wrt x should equal w (elementwise)
	// - gradient wrt w should equal x (elementwise)
	// - gradient wrt b2 (scalar) should be sum of output grads (ones -> 2)
	EXPECT_NEAR(x->GetGradient().m_pValues[0], w->GetValue().m_pValues[0], 1e-6f);
	EXPECT_NEAR(x->GetGradient().m_pValues[1], w->GetValue().m_pValues[1], 1e-6f);

	EXPECT_NEAR(w->GetGradient().m_pValues[0], x->GetValue().m_pValues[0], 1e-6f);
	EXPECT_NEAR(w->GetGradient().m_pValues[1], x->GetValue().m_pValues[1], 1e-6f);

	EXPECT_NEAR(b2->GetGradient().m_pValues[0], 2.0f, 1e-6f);
}



TEST(Neuron2DTests, ForwardAndBackpropTanh)
{ // Create a 2-input neuron with Tanh activation and overwrite weights/bias
	Neuron2D n(2, Activation::Tanh);

	// deterministic weights/bias
	auto w = n.GetWeight();
	auto b = n.GetBias();
	w->value.m_pValues[0] = 1.0f;
	w->value.m_pValues[1] = 0.5f;
	b->value.m_pValues[0] = 1.0f;

	auto x = Node<VectorType>::Create({ 0.5f, 0.0f });

	auto out = n.Forward(x); // returns a length-1 vector (pre_act -> tanh)
	out->Backwards();

	// compute expected values
	// z = w * x -> [0.5, 0.0], sum = 0.5
	// pre = sum + bias = 1.5
	// y = tanh(1.5)
	float pre = 0.5f + 1.0f;
	float y = std::tanh(pre);
	float dy_dpre = 1.0f - y * y;

	// expected gradients
	// bias grad = dy/dpre
	// weight grad_i = input_i * dy/dpre
	// input grad_i = weight_i * dy/dpre
	EXPECT_NEAR(b->GetGradient().m_pValues[0], dy_dpre, 1e-5f);

	EXPECT_NEAR(w->GetGradient().m_pValues[0], x->GetValue().m_pValues[0] * dy_dpre, 1e-5f);
	EXPECT_NEAR(w->GetGradient().m_pValues[1], x->GetValue().m_pValues[1] * dy_dpre, 1e-5f);

	EXPECT_NEAR(x->GetGradient().m_pValues[0], w->GetValue().m_pValues[0] * dy_dpre, 1e-5f);
	EXPECT_NEAR(x->GetGradient().m_pValues[1], w->GetValue().m_pValues[1] * dy_dpre, 1e-5f);
}



TEST(Examples, Kapathy_Example2)
{
	auto x1 = Node<ScalarType>::Create(2.0f, "x1");
	auto x2 = Node<ScalarType>::Create(0.0f, "x2");

	auto w1 = Node<ScalarType>::Create(-3.0f, "w1");
	auto w2 = Node<ScalarType>::Create(1.0f, "w2");

	auto b = Node<ScalarType>::Create(6.881375f, "b");

	auto x1w1 = x1 * w1;
	auto x2w2 = x2 * w2;
	x1w1->SetLabel("x1w1");
	x2w2->SetLabel("x2w2");

	auto x1w1x2w2 = x1w1 + x2w2;
	x1w1x2w2->SetLabel("x1w1x2w2");

	auto n = x1w1x2w2 + b;
	auto o = n->Tanh();

	n->SetLabel("n");
	o->SetLabel("o");

	o->Backwards();

	EXPECT_NEAR(x1->GetValue()[0],     2.0f, 1e-5f);
	EXPECT_NEAR(x1->GetGradient()[0], -1.5f, 1e-5f);

	EXPECT_NEAR(x2->GetValue()[0],    0.0f, 1e-5f);
	EXPECT_NEAR(x2->GetGradient()[0], 0.5f, 1e-5f);

	EXPECT_NEAR(w1->GetValue()[0],   -3.0f, 1e-5f);
	EXPECT_NEAR(w1->GetGradient()[0], 1.0f, 1e-5f);

	EXPECT_NEAR(w2->GetValue()[0],    1.0f, 1e-5f);
	EXPECT_NEAR(w2->GetGradient()[0], 0.0f, 1e-5f);

	EXPECT_NEAR(x1w1->GetValue()[0],   -6.0f, 1e-5f);
	EXPECT_NEAR(x1w1->GetGradient()[0], 0.5f, 1e-5f);

	EXPECT_NEAR(x2w2->GetValue()[0],    0.0f, 1e-5f);
	EXPECT_NEAR(x2w2->GetGradient()[0], 0.5f, 1e-5f);

	EXPECT_NEAR(x1w1x2w2->GetValue()[0],   -6.0f, 1e-5f);
	EXPECT_NEAR(x1w1x2w2->GetGradient()[0], 0.5f, 1e-5f);

	EXPECT_NEAR(b->GetValue()[0], 6.8814f, 1e-4f);
	EXPECT_NEAR(b->GetGradient()[0], 0.5f, 1e-5f);

	EXPECT_NEAR(n->GetValue()[0], 0.8814f, 1e-4f);
	EXPECT_NEAR(n->GetGradient()[0], 0.5f, 1e-5f);

	EXPECT_NEAR(o->GetValue()[0],    0.7071f, 1e-5f);
	EXPECT_NEAR(o->GetGradient()[0], 1.0f, 1e-5f);
}



TEST(Examples, Kapathy_Example2_Vec)
{
	auto x = Node<VectorType>::Create({ 2.0f, 0.0f },  "x");
	auto w = Node<VectorType>::Create({ -3.0f, 1.0f }, "w");
	auto b = Node<VectorType>::Create({ 6.881375f },   "b");

	auto xw = x * w;
	xw->SetLabel("xw");

	auto x1w1x2w2 = xw->Sum();
	x1w1x2w2->SetLabel("x1w1x2w2");

	auto n = x1w1x2w2->ElementwiseAdd(b);
	auto o = n->Tanh();

	n->SetLabel("n");
	o->SetLabel("o");

	o->Backwards();

	EXPECT_NEAR(x->GetValue()[0], 2.0f, 1e-5f);
	EXPECT_NEAR(x->GetGradient()[0], -1.5f, 1e-5f);

	EXPECT_NEAR(x->GetValue()[1], 0.0f, 1e-5f);
	EXPECT_NEAR(x->GetGradient()[1], 0.5f, 1e-5f);

	EXPECT_NEAR(w->GetValue()[0], -3.0f, 1e-5f);
	EXPECT_NEAR(w->GetGradient()[0], 1.0f, 1e-5f);

	EXPECT_NEAR(w->GetValue()[1], 1.0f, 1e-5f);
	EXPECT_NEAR(w->GetGradient()[1], 0.0f, 1e-5f);

	EXPECT_NEAR(xw->GetValue()[0], -6.0f, 1e-5f);
	EXPECT_NEAR(xw->GetGradient()[0], 0.5f, 1e-5f);

	EXPECT_NEAR(xw->GetValue()[1], 0.0f, 1e-5f);
	EXPECT_NEAR(xw->GetGradient()[1], 0.5f, 1e-5f);

	EXPECT_NEAR(x1w1x2w2->GetValue()[0], -6.0f, 1e-5f);
	EXPECT_NEAR(x1w1x2w2->GetGradient()[0], 0.5f, 1e-5f);

	EXPECT_NEAR(b->GetValue()[0], 6.8814f, 1e-4f);
	EXPECT_NEAR(b->GetGradient()[0], 0.5f, 1e-5f);

	EXPECT_NEAR(n->GetValue()[0], 0.8814f, 1e-4f);
	EXPECT_NEAR(n->GetGradient()[0], 0.5f, 1e-5f);

	EXPECT_NEAR(o->GetValue()[0], 0.7071f, 1e-5f);
	EXPECT_NEAR(o->GetGradient()[0], 1.0f, 1e-5f);
}



TEST(Examples, Kapathy_Example2_Neuron)
{
	auto x = Node<VectorType>::Create({ 2.0f, 0.0f }, "x");
	Neuron2D n(2, Activation::Tanh);
	n.GetWeight()->value.m_pValues[0] = -3.0f;
	n.GetWeight()->value.m_pValues[1] = 1.0f;
	n.GetBias()->value.m_pValues[0]   = 6.881375f;

	auto o = n.Forward(x);

	o->Backwards();

	EXPECT_NEAR(x->GetValue()[0], 2.0f, 1e-5f);
	EXPECT_NEAR(x->GetGradient()[0], -1.5f, 1e-5f);

	EXPECT_NEAR(x->GetValue()[1], 0.0f, 1e-5f);
	EXPECT_NEAR(x->GetGradient()[1], 0.5f, 1e-5f);

	EXPECT_NEAR(n.GetWeight()->GetValue()[0], -3.0f, 1e-5f);
	EXPECT_NEAR(n.GetWeight()->GetGradient()[0], 1.0f, 1e-5f);

	EXPECT_NEAR(n.GetWeight()->GetValue()[1], 1.0f, 1e-5f);
	EXPECT_NEAR(n.GetWeight()->GetGradient()[1], 0.0f, 1e-5f);

	EXPECT_NEAR(n.GetBias()->GetValue()[0], 6.8814f, 1e-4f);
	EXPECT_NEAR(n.GetBias()->GetGradient()[0], 0.5f, 1e-5f);

	EXPECT_NEAR(o->GetValue()[0], 0.7071f, 1e-5f);
	EXPECT_NEAR(o->GetGradient()[0], 1.0f, 1e-5f);
}



TEST(Autograd, DiamondGraphGradient)
{
	auto a = Node<ScalarType>::Create(1.0f);

	auto b = a * Node<ScalarType>::Create(2.0f);
	auto c = a * Node<ScalarType>::Create(3.0f);
	auto d = b + c;

	d->Backwards();

	EXPECT_NEAR(a->GetGradient()[0], 5.0f, 1e-5f);
}



TEST(Autograd, SharedNodeMultiply)
{
	auto x = Node<ScalarType>::Create(3.0f);

	auto y = x * x;

	y->Backwards();

	EXPECT_NEAR(x->GetGradient()[0], 6.0f, 1e-5f);
}



TEST(Autograd, VectorMultiplyGradient)
{
	auto a = Node<VectorType>::Create({ 2.0f, 3.0f });
	auto b = Node<VectorType>::Create({ 4.0f, 5.0f });

	auto c = a * b;
	auto loss = c->Sum();

	loss->Backwards();

	EXPECT_NEAR(a->GetGradient()[0], 4.0f, 1e-5f);
	EXPECT_NEAR(a->GetGradient()[1], 5.0f, 1e-5f);

	EXPECT_NEAR(b->GetGradient()[0], 2.0f, 1e-5f);
	EXPECT_NEAR(b->GetGradient()[1], 3.0f, 1e-5f);
}


TEST(Autograd, SumGradient)
{
	auto x = Node<VectorType>::Create({ 1.0f, 2.0f, 3.0f });

	auto y = x->Sum();

	y->Backwards();

	EXPECT_NEAR(x->GetGradient()[0], 1.0f, 1e-5f);
	EXPECT_NEAR(x->GetGradient()[1], 1.0f, 1e-5f);
	EXPECT_NEAR(x->GetGradient()[2], 1.0f, 1e-5f);
}



TEST(Autograd, NeuronGradient)
{
	auto input = Node<VectorType>::Create({ 1.0f, 2.0f });
	auto weight = Node<VectorType>::Create({ 3.0f, 4.0f });
	auto bias = Node<VectorType>::Create({ 5.0f });

	auto z = weight * input;
	auto s = z->Sum();
	auto p = s + bias;

	p->Backwards();

	EXPECT_NEAR(weight->GetGradient()[0], 1.0f, 1e-5f);
	EXPECT_NEAR(weight->GetGradient()[1], 2.0f, 1e-5f);

	EXPECT_NEAR(input->GetGradient()[0], 3.0f, 1e-5f);
	EXPECT_NEAR(input->GetGradient()[1], 4.0f, 1e-5f);

	EXPECT_NEAR(bias->GetGradient()[0], 1.0f, 1e-5f);
}



TEST(Autograd, TanhGradient)
{
	auto x = Node<ScalarType>::Create(0.5f);

	auto y = x->Tanh();

	y->Backwards();

	float expected = 1.0f - std::tanh(0.5f) * std::tanh(0.5f);

	EXPECT_NEAR(x->GetGradient()[0], expected, 1e-5f);
}



TEST(Training, GradientChangesAfterUpdate)
{
	auto x = Node<VectorType>::Create({ 1.0f, 2.0f });
	auto target = Node<VectorType>::Create({ 1.0f });

	Neuron2D n(2, Activation::Tanh);

	// deterministic params
	n.GetWeight()->value.m_pValues[0] = 0.5f;
	n.GetWeight()->value.m_pValues[1] = -0.3f;
	n.GetBias()->value.m_pValues[0] = 0.1f;

	// --- first pass ---
	auto out1 = n.Forward(x);

	auto diff1 = out1 - target;
	auto loss1 = diff1 * diff1;

	loss1->Backwards();

	float grad_before = n.GetWeight()->GetGradient()[0];

	// SGD step
	n.GetWeight()->value.m_pValues[0] -= 0.01f * grad_before;

	// --- second pass ---
	auto out2 = n.Forward(x);

	auto diff2 = out2 - target;
	auto loss2 = diff2 * diff2;

	loss2->Backwards();

	float grad_after = n.GetWeight()->GetGradient()[0];

	// gradients should differ after parameter update
	EXPECT_NE(grad_before, grad_after);
}



TEST(Node, ValueIsMutable)
{
	auto n = Node<ScalarType>::Create(2.0f);

	auto before = n->GetValue()[0];

	n->GetValue() -= 1.0f;

	auto after = n->GetValue()[0];

	EXPECT_NE(before, after);
}



TEST(Node, GradientIsMutable)
{
	auto n = Node<ScalarType>::Create(2.0f);

	auto before = n->GetGradient()[0];

	n->GetGradient() -= 1.0f;

	auto after = n->GetGradient()[0];

	EXPECT_NE(before, after);
}



TEST(Autograd, Pack_NoDoubleGradientPropagation)
{
	auto x1 = Node<VectorType>::Create({ 1.0f });
	auto x2 = Node<VectorType>::Create({ 2.0f });
	auto x3 = Node<VectorType>::Create({ 3.0f });

	std::vector<NodePtr<VectorType>> inputs = { x1, x2, x3 };

	auto packed = Node<VectorType>::CreateWithSize(3);
	packed->Pack(inputs);

	EXPECT_NEAR(packed->GetValue()[0], x1->GetValue()[0], 1e-5f);
	EXPECT_NEAR(packed->GetValue()[1], x2->GetValue()[0], 1e-5f);
	EXPECT_NEAR(packed->GetValue()[2], x3->GetValue()[0], 1e-5f);

	// loss = sum of packed vector
	auto loss = packed->Sum();

	EXPECT_NEAR(loss->GetValue()[0], x1->GetValue()[0] + x2->GetValue()[0] + x3->GetValue()[0], 1e-5f);

	loss->Backwards();

	// expected gradient = 1 for each input exactly once
	EXPECT_NEAR(x1->GetGradient()[0], 1.0f, 1e-5f);
	EXPECT_NEAR(x2->GetGradient()[0], 1.0f, 1e-5f);
	EXPECT_NEAR(x3->GetGradient()[0], 1.0f, 1e-5f);
}



TEST(Autograd, DetectDoubleBackpropagation)
{
	// shared node
	auto x = Node<ScalarType>::Create(2.0f);

	// diamond graph with shared dependency
	auto a = x * Node<ScalarType>::Create(3.0f);
	auto b = x * Node<ScalarType>::Create(4.0f);

	auto y = a + b;

	y->Backwards();

	// mathematically:
	// y = x*3 + x*4 = x*(3+4) = 7x
	// dy/dx = 7

	EXPECT_NEAR(x->GetGradient()[0], 7.0f, 1e-5f);
}



TEST(Autograd, DetectMultipleBackpropPathsExplosion)
{
	auto x = Node<ScalarType>::Create(1.0f);

	auto a = x * Node<ScalarType>::Create(2.0f);
	auto b = x * Node<ScalarType>::Create(3.0f);
	auto c = x * Node<ScalarType>::Create(4.0f);

	auto y = a + b + c;

	y->Backwards();

	// correct:
	// dy/dx = 2 + 3 + 4 = 9

	EXPECT_NEAR(x->GetGradient()[0], 9.0f, 1e-5f);
}



TEST(Training, XOR_OverfitSingleEpochSanity)
{
	auto x = Node<VectorType>::Create({ 0.0f, 1.0f });
	auto y_true = Node<VectorType>::Create({ 1.0f });

	Neuron2D n(2, Activation::Tanh);

	// make test deterministic
	n.GetWeight()->value.m_pValues[0] = 0.5f;
	n.GetWeight()->value.m_pValues[1] = -0.3f;
	n.GetBias()->value.m_pValues[0] = 0.1f;

	auto out = n.Forward(x);

	auto diff = out - y_true;
	auto loss = diff * diff;

	loss->Backwards();

	// forward sanity (should never change)
	EXPECT_TRUE(std::isfinite(out->GetValue()[0]));

	// compute expected gradient manually
	float z = 0.5f * 0.0f + (-0.3f) * 1.0f + 0.1f;
	float y = std::tanh(z);

	float dL_dy = 2.0f * (y - 1.0f);
	float dL_dz = dL_dy * (1.0f - y * y);

	// only w1 should receive gradient (x = [0,1])
	EXPECT_NEAR(n.GetWeight()->GetGradient()[0], 0.0f, 1e-6f);
	EXPECT_NEAR(n.GetWeight()->GetGradient()[1], dL_dz * 1.0f, 1e-6f);

	// bias always receives gradient
	EXPECT_NEAR(n.GetBias()->GetGradient()[0], dL_dz, 1e-6f);

	// gradient must be valid and non-zero (since we fixed params)
	EXPECT_TRUE(std::isfinite(n.GetWeight()->GetGradient()[1]));
	EXPECT_NE(n.GetWeight()->GetGradient()[1], 0.0f);
}



TEST(Training, XOR)
{
	std::vector<NodePtr<VectorType>> xs = {
		Node<VectorType>::Create({ 0.0f, 0.0f }),
		Node<VectorType>::Create({ 0.0f, 1.0f }),
		Node<VectorType>::Create({ 1.0f, 0.0f }),
		Node<VectorType>::Create({ 1.0f, 1.0f })
	};

	std::vector<NodePtr<VectorType>> label = {
		Node<VectorType>::Create({ 0.0f }),
		Node<VectorType>::Create({ 1.0f }),
		Node<VectorType>::Create({ 1.0f }),
		Node<VectorType>::Create({ 0.0f })
	};

	Layer2D<VectorType> L1(2, 3, Activation::Tanh);
	Layer2D<VectorType> L2(3, 1, Activation::Tanh);

	float lr = 0.1f;
	float epoch_loss = 0.0f;

	for (size_t epoch = 0; epoch < 200; epoch++)
	{
		epoch_loss = 0.0f;

		for (size_t i = 0; i < xs.size(); i++)
		{
			auto x = xs[i];

			auto l1_out = L1.Forward(x);
			auto out = L2.Forward(l1_out);

			auto diff = out - label[i];
			auto loss = diff * diff;

			auto params = loss->CollectParams();
			EXPECT_EQ(params.size(), (3 + 1) * 2);//3 + 1 neurons with weight and bias

			loss->Backwards();

			for (size_t i = 0; i < L1.GetOutputLength(); i++)
			{
				L1.GetWeight(i)->GetValue() -= lr * L1.GetWeight(i)->GetGradient();
				L1.GetBias(i)->GetValue()   -= lr * L1.GetBias(i)->GetGradient();
			}
			for (size_t i = 0; i < L2.GetOutputLength(); i++)
			{
				L2.GetWeight(i)->GetValue() -= lr * L2.GetWeight(i)->GetGradient();
				L2.GetBias(i)->GetValue()   -= lr * L2.GetBias(i)->GetGradient();
			}

			epoch_loss += loss->GetValue()[0];
		}
	}

	EXPECT_NEAR(epoch_loss, 0.0f, 0.1f);

	for (size_t i = 0; i < xs.size(); i++)
	{
		auto x = xs[i];

		auto l1_out = L1.Forward(x);
		auto out = L2.Forward(l1_out);

		auto diff = out - label[i];
		EXPECT_NEAR(diff->GetValue()[0], 0.0f, 0.1f);
	}
}



TEST(Autograd, ScalarLossSanity)
{
	auto x = Node<VectorType>::Create({ 0.5f });
	auto y = Node<VectorType>::Create({ 1.0f });

	auto diff = x - y;
	auto loss = (diff * diff)->Sum();

	loss->Backwards();

	EXPECT_NEAR(loss->GetValue()[0], 0.25f, 1e-5f);
	EXPECT_NEAR(x->GetGradient()[0], -1.0f, 1e-5f);
}



TEST(SGD, XOR)
{
	std::vector<NodePtr<VectorType>> xs = {
		Node<VectorType>::Create({ 0.0f, 0.0f }),
		Node<VectorType>::Create({ 0.0f, 1.0f }),
		Node<VectorType>::Create({ 1.0f, 0.0f }),
		Node<VectorType>::Create({ 1.0f, 1.0f })
	};

	std::vector<NodePtr<VectorType>> label = {
		Node<VectorType>::Create({ 0.0f }),
		Node<VectorType>::Create({ 1.0f }),
		Node<VectorType>::Create({ 1.0f }),
		Node<VectorType>::Create({ 0.0f })
	};

	Layer2D<VectorType> L1(2, 3, Activation::Tanh);
	Layer2D<VectorType> L2(3, 1, Activation::Tanh);

	SGD<VectorType> sgd(0.1f);

	auto l1_out = L1.Forward(xs[0]);
	auto out = L2.Forward(l1_out);

	auto diff = out - label[0];
	auto loss = diff * diff;

	sgd.SetTrainableParams(loss->CollectParams());

	float lr = 0.1f;
	float epoch_loss = 0.0f;

	for (size_t epoch = 0; epoch < 300; epoch++)
	{
		epoch_loss = 0.0f;

		for (size_t i = 0; i < xs.size(); i++)
		{
			auto x = xs[i];

			auto l1_out = L1.Forward(x);
			auto out = L2.Forward(l1_out);

			auto diff = out - label[i];
			auto loss = diff * diff;

			loss->Backwards();

			sgd.Step();

			epoch_loss += loss->GetValue()[0];
		}
	}

	EXPECT_NEAR(epoch_loss, 0.0f, 0.1f);

	for (size_t i = 0; i < xs.size(); i++)
	{
		auto x = xs[i];

		auto l1_out = L1.Forward(x);
		auto out = L2.Forward(l1_out);

		auto diff = out - label[i];
		EXPECT_NEAR(diff->GetValue()[0], 0.0f, 0.1f);
	}
}



TEST(SGD, XOR_BATCHED)
{
	std::vector<NodePtr<VectorType>> xs = {
		Node<VectorType>::Create({ 0.0f, 0.0f }),
		Node<VectorType>::Create({ 0.0f, 1.0f }),
		Node<VectorType>::Create({ 1.0f, 0.0f }),
		Node<VectorType>::Create({ 1.0f, 1.0f })
	};

	std::vector<NodePtr<VectorType>> labels = {
		Node<VectorType>::Create({ 0.0f }),
		Node<VectorType>::Create({ 1.0f }),
		Node<VectorType>::Create({ 1.0f }),
		Node<VectorType>::Create({ 0.0f })
	};

	Layer2D<VectorType> L1(2, 3, Activation::Tanh);
	Layer2D<VectorType> L2(3, 1, Activation::Tanh);

	SGD<VectorType> sgd(0.1f);

	const int batch_size = 4; // XOR fits perfectly in one batch

	// IMPORTANT: collect params ONCE from full graph
	{
		auto x    = xs[0];
		auto h1   = L1.Forward(x);
		auto out  = L2.Forward(h1);
		auto loss = (out - labels[0]) * (out - labels[0]);

		sgd.SetTrainableParams(loss->CollectParams());
	}

	float epoch_loss = 0.0f;

	for (size_t epoch = 0; epoch < 300; epoch++)
	{
		epoch_loss = 0.0f;

		// iterate in batches
		for (size_t i = 0; i < xs.size(); i += batch_size)
		{
			size_t end = std::min(i + batch_size, xs.size());

			NodePtr<VectorType> batch_loss = nullptr;

			for (size_t j = i; j < end; j++)
			{
				auto x = xs[j];

				auto h1  = L1.Forward(x);
				auto out = L2.Forward(h1);

				auto diff = out - labels[j];
				auto loss = diff * diff;

				epoch_loss += loss->GetValue()[0];

				// accumulate batch loss
				if (!batch_loss)
					batch_loss = loss;
				else
					batch_loss = batch_loss + loss;
			}

			// backward ONCE per batch
			batch_loss->Backwards();

			// update once per batch
			sgd.Step();
		}

		epoch_loss /= xs.size();

		if (epoch % 20 == 0)
			std::cout << "Epoch " << epoch << " Loss: " << epoch_loss << std::endl;
	}

	// -----------------------
	// Final correctness check
	// -----------------------
	for (size_t i = 0; i < xs.size(); i++)
	{
		auto h1 = L1.Forward(xs[i]);
		auto out = L2.Forward(h1);

		float pred = out->GetValue()[0];
		float target = labels[i]->GetValue()[0];

		std::cout << "xored pred=" << pred
			<< " target=" << target << std::endl;

		EXPECT_NEAR(pred, target, 0.2f);
	}
}



TEST(SGD, SinRegression)
{
	// training data: y = sin(x)
	std::vector<NodePtr<VectorType>> xs;
	std::vector<NodePtr<VectorType>> labels;

	const int N = 20;

	for (int i = 0; i < N; i++)
	{
		float x = -3.1415f + i * (6.2830f / (N - 1)); // [-pi, pi]
		float y = std::sin(x);

		xs.push_back(Node<VectorType>::Create({ x }));
		labels.push_back(Node<VectorType>::Create({ y }));
	}

	// simple MLP: 1 → 8 → 8 → 1
	Layer2D<VectorType> L1(1, 8, Activation::Tanh);
	Layer2D<VectorType> L2(8, 8, Activation::Tanh);
	Layer2D<VectorType> L3(8, 1, Activation::Tanh);

	SGD<VectorType> sgd(0.02f);

	float epoch_loss = 0.0f;

	for (size_t epoch = 0; epoch < 800; epoch++)
	{
		epoch_loss = 0.0f;

		for (size_t i = 0; i < xs.size(); i++)
		{
			auto x = xs[i];

			auto h1 = L1.Forward(x);
			auto h2 = L2.Forward(h1);
			auto pred = L3.Forward(h2);

			auto diff = pred - labels[i];
			auto loss = diff * diff;

			sgd.SetTrainableParams(loss->CollectParams());

			loss->Backwards();
			sgd.Step();

			epoch_loss += loss->GetValue()[0];
		}

		epoch_loss /= xs.size();

		// optional debug
		if (epoch % 100 == 0)
			std::cout << "Epoch " << epoch << " Loss: " << epoch_loss << std::endl;
	}

	// final sanity check: model learned function shape
	for (size_t i = 0; i < xs.size(); i++)
	{
		auto x = xs[i];

		auto h1  = L1.Forward(x);
		auto h2  = L2.Forward(h1);
		auto out = L3.Forward(h2);

		float pred = out->GetValue()[0];
		float target = labels[i]->GetValue()[0];

		EXPECT_NEAR(pred, target, 0.2f);
	}
}



TEST(SGD, SpiralClassification_MSE)
{
	std::vector<NodePtr<VectorType>> xs;
	std::vector<NodePtr<VectorType>> labels;

	const int points_per_class = 50;
	const float pi = 3.14159265f;

	// generate 2-class spiral dataset
	for (int class_id = 0; class_id < 2; class_id++)
	{
		for (int i = 0; i < points_per_class; i++)
		{
			float r = (float)i / points_per_class;

			// spiral angle
			float t = 4.0f * pi * r + class_id * pi;

			float x = r * std::sin(t);
			float y = r * std::cos(t);

			xs.push_back(Node<VectorType>::Create({ x, y }));

			// one-hot-ish target for MSE
			if (class_id == 0)
				labels.push_back(Node<VectorType>::Create({ 0.0f }));
			else
				labels.push_back(Node<VectorType>::Create({ 1.0f }));
		}
	}

	// MLP: 2 -> 32 -> 32 -> 1
	Layer2D<VectorType> L1(2, 32, Activation::Tanh);
	Layer2D<VectorType> L2(32, 32, Activation::Tanh);
	Layer2D<VectorType> L3(32, 1, Activation::Tanh);

	SGD<VectorType> sgd(0.01f);
	const int batch_size = 8;

	// IMPORTANT: collect params ONCE
	{
		auto x = xs[0];
		auto h1 = L1.Forward(x);
		auto h2 = L2.Forward(h1);
		auto out = L3.Forward(h2);

		auto loss = (out - labels[0]) * (out - labels[0]);
		sgd.SetTrainableParams(loss->CollectParams());
	}

	//Training

	float epoch_loss = 0.0f;

	for (size_t epoch = 0; epoch < 1200; epoch++)
	{
		ShuffleDataset(xs, labels);

		epoch_loss = 0.0f;

		for (size_t i = 0; i < xs.size(); i++)
		{
			auto x = xs[i];

			auto h1 = L1.Forward(x);
			auto h2 = L2.Forward(h1);
			auto out = L3.Forward(h2);

			auto diff = out - labels[i];
			auto loss = diff * diff;

			sgd.SetTrainableParams(loss->CollectParams());

			loss->Backwards();
			sgd.Step();

			epoch_loss += loss->GetValue()[0];
		}

		epoch_loss /= xs.size();

		if (epoch % 100 == 0)
			std::cout << "Epoch " << epoch << " Loss: " << epoch_loss << std::endl;
	}

	// evaluate classification accuracy
	int correct = 0;

	for (size_t i = 0; i < xs.size(); i++)
	{
		auto h1 = L1.Forward(xs[i]);
		auto h2 = L2.Forward(h1);
		auto out = L3.Forward(h2);

		float pred = out->GetValue()[0];
		float target = labels[i]->GetValue()[0];

		int predicted_class = pred > 0.5f ? 1 : 0;
		int target_class = target > 0.5f ? 1 : 0;

		if (predicted_class == target_class)
			correct++;
	}

	float accuracy = (float)correct / xs.size();

	std::cout << "Final Accuracy: " << accuracy << std::endl;

	EXPECT_GT(accuracy, 0.90f);
}