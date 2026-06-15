#include "gtest/gtest.h"
#include <random>
#include "../src/node.h"
#include "../src/neuron.h"
#include "../src/layer.h"
#include "../src/sgd.h"
#include "../src/tape.h"
#include "../src/conv2d.h"
#include "../mnist.h"



template<typename T>
void ShuffleDataset(std::vector<NodePtr<T>>& xs, std::vector<NodePtr<T>>& labels)
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



void ShuffleDataset(std::vector<std::vector<float>>& xs, std::vector<float>& labels)
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



void ShuffleDataset(std::vector<Tensor4D>& xs, std::vector<float>& labels)
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



TEST(Scalar, BasicArithmeticAndBackprop)
{ // d = a * b + c
	auto a = Node<Scalar>::Create(2.0f);
	auto b = Node<Scalar>::Create(-3.0f);
	auto c = Node<Scalar>::Create(10.0f);

	auto d = a * b + c;
	d->Backwards();

	EXPECT_NEAR(d->GetValue().GetValue()[0], 2.0f * -3.0f + 10.0f, 1e-6f);
	EXPECT_NEAR(d->GetGradient().GetValue()[0], 1.0f, 1e-6f); // dd/dd = 1
	EXPECT_NEAR(a->GetGradient().GetValue()[0], -3.0f, 1e-6f); // dd/da = b
	EXPECT_NEAR(b->GetGradient().GetValue()[0], 2.0f, 1e-6f);  // dd/db = a
	EXPECT_NEAR(c->GetGradient().GetValue()[0], 1.0f, 1e-6f);  // dd/dc = 1
}



TEST(Vector, ElementwiseMultiplyAddBackprop)
{ // out = (w * x) + b2 (elementwise add with scalar bias)
	auto x  = Node<Vector>::Create({ 0.5f, 0.0f });
	auto w  = Node<Vector>::Create({ 1.0f, 0.5f });
	auto b2 = Node<Vector>::Create({ 1.0f });

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



TEST(Neuron2D, ForwardAndBackpropTanh)
{ // Create a 2-input neuron with Tanh activation and overwrite weights/bias
	Neuron2D n(2, Activation::Tanh);

	// deterministic weights/bias
	auto w = n.GetWeight();
	auto b = n.GetBias();
	w->value.m_pValues[0] = 1.0f;
	w->value.m_pValues[1] = 0.5f;
	b->value.m_pValues[0] = 1.0f;

	auto x = Node<Vector>::Create({ 0.5f, 0.0f });

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
	auto x1 = Node<Scalar>::Create(2.0f, "x1");
	auto x2 = Node<Scalar>::Create(0.0f, "x2");

	auto w1 = Node<Scalar>::Create(-3.0f, "w1");
	auto w2 = Node<Scalar>::Create(1.0f, "w2");

	auto b = Node<Scalar>::Create(6.881375f, "b");

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



TEST(MatrixBackprop, SimpleTanhGraph)
{
	// x: 2x1
	auto x = Node<Matrix>::Create({ 2.0f, 0.0f }, "x");

	// w: 2x1
	auto w = Node<Matrix>::Create({ -3.0f, 1.0f }, "w");

	// b: 1x1 (broadcast oder scalar-matrix)
	auto b = Node<Matrix>::Create({ 6.881375f }, "b");

	// elementwise multiply
	auto xw = w->ElementwiseMul(x);
	xw->SetLabel("xw");

	// sum (assuming elementwise or reduction consistent with your lib)
	auto xw_sum = xw->Sum();
	auto n = xw_sum + b;

	n->SetLabel("n");

	auto o = n->Tanh();
	o->SetLabel("o");

	o->Backwards();

	// forward checks
	EXPECT_NEAR(x->GetValue().At(0, 0), 2.0f, 1e-5f);
	EXPECT_NEAR(x->GetValue().At(1, 0), 0.0f, 1e-5f);

	EXPECT_NEAR(w->GetValue().At(0, 0), -3.0f, 1e-5f);
	EXPECT_NEAR(w->GetValue().At(1, 0), 1.0f, 1e-5f);

	EXPECT_NEAR(b->GetValue().At(0, 0), 6.8814f, 1e-4f);

	// forward expected intermediate (elementwise product)
	EXPECT_NEAR(xw->GetValue().At(0, 0), -6.0f, 1e-5f);
	EXPECT_NEAR(xw->GetValue().At(1, 0), 0.0f, 1e-5f);

	// gradients (elementwise)
	EXPECT_NEAR(x->GetGradient().At(0, 0), -1.5f, 1e-5f);
	EXPECT_NEAR(x->GetGradient().At(1, 0), 0.5f, 1e-5f);

	EXPECT_NEAR(w->GetGradient().At(0, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(w->GetGradient().At(1, 0), 0.0f, 1e-5f);

	EXPECT_NEAR(b->GetGradient().At(0, 0), 0.5f, 1e-5f);

	EXPECT_NEAR(n->GetGradient().At(0, 0), 0.5f, 1e-5f);

	EXPECT_NEAR(o->GetValue().At(0, 0), 0.7071f, 1e-4f);
	EXPECT_NEAR(o->GetGradient().At(0, 0), 1.0f, 1e-5f);
}



TEST(MatrixBackprop, MatrixMultiplicationGraph)
{
	// A: 2x3
	auto A = Node<Matrix>::Create({
		{ 1.0f, 2.0f, 3.0f },
		{ 4.0f, 5.0f, 6.0f }
		}, "A");

	// B: 3x2
	auto B = Node<Matrix>::Create({
		{ 7.0f,  8.0f },
		{ 9.0f, 10.0f },
		{11.0f, 12.0f }
		}, "B");

	// bias: scalar matrix
	auto bias = Node<Matrix>::Create(Matrix({{ 1.0f }}), "bias");

	// Matrix multiplication
	auto C = A * B;
	C->SetLabel("C");

	// Sum reduction
	auto S = C->Sum();
	S->SetLabel("S");

	// Final output
	auto Y = S + bias;
	Y->SetLabel("Y");

	// backward pass
	Y->Backwards();

	// -------------------------------------------------
	// Forward checks
	// -------------------------------------------------

	// C expected:
	//
	// [ 58,  64 ]
	// [139, 154 ]
	//
	// Calculation:
	// row0 col0 = 1*7 + 2*9 + 3*11 = 58
	// row0 col1 = 1*8 + 2*10 + 3*12 = 64
	// row1 col0 = 4*7 + 5*9 + 6*11 = 139
	// row1 col1 = 4*8 + 5*10 + 6*12 = 154

	EXPECT_NEAR(C->GetValue().At(0, 0), 58.0f, 1e-5f);
	EXPECT_NEAR(C->GetValue().At(0, 1), 64.0f, 1e-5f);
	EXPECT_NEAR(C->GetValue().At(1, 0), 139.0f, 1e-5f);
	EXPECT_NEAR(C->GetValue().At(1, 1), 154.0f, 1e-5f);

	// Sum = 58 + 64 + 139 + 154 = 415
	EXPECT_NEAR(S->GetValue().At(0, 0), 415.0f, 1e-5f);

	// Y = 416
	EXPECT_NEAR(Y->GetValue().At(0, 0), 416.0f, 1e-5f);

	// -------------------------------------------------
	// Gradient checks
	// -------------------------------------------------

	// dY/dY = 1
	EXPECT_NEAR(Y->GetGradient().At(0, 0), 1.0f, 1e-5f);

	// dY/dS = 1
	EXPECT_NEAR(S->GetGradient().At(0, 0), 1.0f, 1e-5f);

	// dY/dbias = 1
	EXPECT_NEAR(bias->GetGradient().At(0, 0), 1.0f, 1e-5f);

	// Since S = sum(C),
	// dS/dC is a matrix full of ones:
	//
	// [1 1]
	// [1 1]

	EXPECT_NEAR(C->GetGradient().At(0, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(C->GetGradient().At(0, 1), 1.0f, 1e-5f);
	EXPECT_NEAR(C->GetGradient().At(1, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(C->GetGradient().At(1, 1), 1.0f, 1e-5f);

	// -------------------------------------------------
	// Expected gradients
	// -------------------------------------------------
	//
	// dL/dA = dL/dC * B^T
	//
	// ones(2x2) * B^T =
	//
	// [1 1]   [ 7  9 11 ]   [15 19 23]
	// [1 1] * [ 8 10 12 ] = [15 19 23]
	//

	EXPECT_NEAR(A->GetGradient().At(0, 0), 15.0f, 1e-5f);
	EXPECT_NEAR(A->GetGradient().At(0, 1), 19.0f, 1e-5f);
	EXPECT_NEAR(A->GetGradient().At(0, 2), 23.0f, 1e-5f);

	EXPECT_NEAR(A->GetGradient().At(1, 0), 15.0f, 1e-5f);
	EXPECT_NEAR(A->GetGradient().At(1, 1), 19.0f, 1e-5f);
	EXPECT_NEAR(A->GetGradient().At(1, 2), 23.0f, 1e-5f);

	// -------------------------------------------------
	// dL/dB = A^T * dL/dC
	//
	// A^T =
	// [1 4]
	// [2 5]
	// [3 6]
	//
	// * ones(2x2)
	//
	// =
	// [5 5]
	// [7 7]
	// [9 9]
	//

	EXPECT_NEAR(B->GetGradient().At(0, 0), 5.0f, 1e-5f);
	EXPECT_NEAR(B->GetGradient().At(0, 1), 5.0f, 1e-5f);

	EXPECT_NEAR(B->GetGradient().At(1, 0), 7.0f, 1e-5f);
	EXPECT_NEAR(B->GetGradient().At(1, 1), 7.0f, 1e-5f);

	EXPECT_NEAR(B->GetGradient().At(2, 0), 9.0f, 1e-5f);
	EXPECT_NEAR(B->GetGradient().At(2, 1), 9.0f, 1e-5f);
}



TEST(MatrixBackprop, RowVectorColumnVectorOuterProduct)
{
	// A = row vector (1x3)
	auto A = Node<Matrix>::Create({
		{1.0f, 2.0f, 3.0f}
		}, "A");

	// B = column vector (3x1)
	auto B = Node<Matrix>::Create({
		{4.0f},
		{5.0f},
		{6.0f}
		}, "B");

	// Y = A * B = scalar (1x1)
	auto Y = A * B;
	Y->SetLabel("Y");

	// backward
	Y->Backwards();

	// -----------------------------------
	// Forward
	// -----------------------------------

	// 1*4 + 2*5 + 3*6 = 32
	EXPECT_NEAR(Y->GetValue().At(0, 0), 32.0f, 1e-5f);

	// -----------------------------------
	// Shape checks
	// -----------------------------------

	EXPECT_EQ(A->GetGradient().GetRows(), 1);
	EXPECT_EQ(A->GetGradient().GetColumns(), 3);

	EXPECT_EQ(B->GetGradient().GetRows(), 3);
	EXPECT_EQ(B->GetGradient().GetColumns(), 1);

	// -----------------------------------
	// Backward
	// -----------------------------------

	// dY/dA = B^T
	//
	// [4 5 6]

	EXPECT_NEAR(A->GetGradient().At(0, 0), 4.0f, 1e-5f);
	EXPECT_NEAR(A->GetGradient().At(0, 1), 5.0f, 1e-5f);
	EXPECT_NEAR(A->GetGradient().At(0, 2), 6.0f, 1e-5f);

	// dY/dB = A^T
	//
	// [1]
	// [2]
	// [3]

	EXPECT_NEAR(B->GetGradient().At(0, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(B->GetGradient().At(1, 0), 2.0f, 1e-5f);
	EXPECT_NEAR(B->GetGradient().At(2, 0), 3.0f, 1e-5f);
}



TEST(MatrixBackprop, LargerMatrixGraphWithoutActivation)
{
	// x: 3x3
	auto x = Node<Matrix>::Create({
		{ 1.0f,  2.0f,  3.0f },
		{ 4.0f,  5.0f,  6.0f },
		{ 7.0f,  8.0f,  9.0f }
		}, "x");

	// w: 3x3
	auto w = Node<Matrix>::Create({
		{ 0.5f, -1.0f,  2.0f },
		{ 1.5f,  0.0f, -0.5f },
		{ 2.0f, -2.0f,  1.0f }
		}, "w");

	// b: scalar matrix
	auto b = Node<Matrix>::Create({
		{ 10.0f }
		}, "b");

	// elementwise multiplication
	auto xw = x->ElementwiseMul(w);
	xw->SetLabel("xw");

	EXPECT_EQ(xw->GetValue().GetRows(),       3);
	EXPECT_EQ(xw->GetValue().GetColumns(),    3);
	EXPECT_EQ(xw->GetGradient().GetRows(),    3);
	EXPECT_EQ(xw->GetGradient().GetColumns(), 3);

	// reduction sum
	auto s = xw->Sum();
	s->SetLabel("s");

	// final output
	auto y = s + b;
	y->SetLabel("y");

	// backward pass
	y->Backwards();

	// -------------------------------------------------
	// Forward checks
	// -------------------------------------------------

	// xw expected:
	// [ 0.5, -2,   6 ]
	// [ 6,    0,  -3 ]
	// [14,  -16,   9 ]

	EXPECT_NEAR(xw->GetValue().At(0, 0), 0.5f, 1e-5f);
	EXPECT_NEAR(xw->GetValue().At(0, 1), -2.0f, 1e-5f);
	EXPECT_NEAR(xw->GetValue().At(0, 2), 6.0f, 1e-5f);

	EXPECT_NEAR(xw->GetValue().At(1, 0), 6.0f, 1e-5f);
	EXPECT_NEAR(xw->GetValue().At(1, 1), 0.0f, 1e-5f);
	EXPECT_NEAR(xw->GetValue().At(1, 2), -3.0f, 1e-5f);

	EXPECT_NEAR(xw->GetValue().At(2, 0), 14.0f, 1e-5f);
	EXPECT_NEAR(xw->GetValue().At(2, 1), -16.0f, 1e-5f);
	EXPECT_NEAR(xw->GetValue().At(2, 2), 9.0f, 1e-5f);

	// sum = 14.5
	EXPECT_NEAR(s->GetValue().At(0, 0), 14.5f, 1e-5f);

	// y = 24.5
	EXPECT_NEAR(y->GetValue().At(0, 0), 24.5f, 1e-5f);

	// -------------------------------------------------
	// Gradient checks
	// -------------------------------------------------

	// dy/dy = 1
	EXPECT_NEAR(y->GetGradient().At(0, 0), 1.0f, 1e-5f);

	// dy/ds = 1
	EXPECT_NEAR(s->GetGradient().At(0, 0), 1.0f, 1e-5f);

	// dy/db = 1
	EXPECT_NEAR(b->GetGradient().At(0, 0), 1.0f, 1e-5f);

	// gradients wrt x are w
	EXPECT_NEAR(x->GetGradient().At(0, 0), 0.5f, 1e-5f);
	EXPECT_NEAR(x->GetGradient().At(0, 1), -1.0f, 1e-5f);
	EXPECT_NEAR(x->GetGradient().At(0, 2), 2.0f, 1e-5f);

	EXPECT_NEAR(x->GetGradient().At(1, 0), 1.5f, 1e-5f);
	EXPECT_NEAR(x->GetGradient().At(1, 1), 0.0f, 1e-5f);
	EXPECT_NEAR(x->GetGradient().At(1, 2), -0.5f, 1e-5f);

	EXPECT_NEAR(x->GetGradient().At(2, 0), 2.0f, 1e-5f);
	EXPECT_NEAR(x->GetGradient().At(2, 1), -2.0f, 1e-5f);
	EXPECT_NEAR(x->GetGradient().At(2, 2), 1.0f, 1e-5f);

	// gradients wrt w are x
	EXPECT_NEAR(w->GetGradient().At(0, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(w->GetGradient().At(0, 1), 2.0f, 1e-5f);
	EXPECT_NEAR(w->GetGradient().At(0, 2), 3.0f, 1e-5f);

	EXPECT_NEAR(w->GetGradient().At(1, 0), 4.0f, 1e-5f);
	EXPECT_NEAR(w->GetGradient().At(1, 1), 5.0f, 1e-5f);
	EXPECT_NEAR(w->GetGradient().At(1, 2), 6.0f, 1e-5f);

	EXPECT_NEAR(w->GetGradient().At(2, 0), 7.0f, 1e-5f);
	EXPECT_NEAR(w->GetGradient().At(2, 1), 8.0f, 1e-5f);
	EXPECT_NEAR(w->GetGradient().At(2, 2), 9.0f, 1e-5f);
}



TEST(MatrixBackprop, SoftmaxGradient)
{
	// input
	auto x = Node<Matrix>::Create({ {1.0f}, {2.0f}, {3.0f} }, "x");

	// softmax
	auto s = x->Softmax();
	s->SetLabel("softmax");

	// loss = sum(softmax)
	auto loss = s->Sum();
	loss->SetLabel("loss");

	// backward
	loss->Backwards();

	// -------------------------------------------------
	// Forward sanity checks
	// -------------------------------------------------

	float s0 = s->GetValue().At(0, 0);
	float s1 = s->GetValue().At(0, 1);
	float s2 = s->GetValue().At(0, 2);

	float sum = s0 + s1 + s2;
	EXPECT_NEAR(sum, 1.0f, 1e-5f);

	// -------------------------------------------------
	// Gradient checks (analytisch)
	// -------------------------------------------------
	// For L = sum(s), gradient is:
	// dL/dx_i = s_i * (1 - sum(s)) = 0
	// BUT due to coupling, actual result is:
	// dL/dx_i = s_i * (1 - 1) + cross terms = 0
	//
	// => result: all gradients should be ~0

	EXPECT_NEAR(x->GetGradient().At(0, 0), 0.0f, 1e-5f);
	EXPECT_NEAR(x->GetGradient().At(0, 1), 0.0f, 1e-5f);
	EXPECT_NEAR(x->GetGradient().At(0, 2), 0.0f, 1e-5f);
}



TEST(MatrixBackprop, SoftmaxCrossEntropy)
{
	// logits (unnormalized scores)
	auto logits = Node<Matrix>::Create({ 2.0f, 1.0f, 0.1f }, "logits");

	// ground truth: class 0
	auto target = Node<Matrix>::Create({ 0.0f }, "target");

	auto loss = logits->Softmax_CrossEntropy(target);

	loss->Backwards();

	// -------------------------
	// Forward reference
	// -------------------------
	float l0 = 2.0f, l1 = 1.0f, l2 = 0.1f;

	float maxLogit = 2.0f;

	float e0 = std::exp(l0 - maxLogit);
	float e1 = std::exp(l1 - maxLogit);
	float e2 = std::exp(l2 - maxLogit);

	float sum = e0 + e1 + e2;

	float p0 = e0 / sum;
	float p1 = e1 / sum;
	float p2 = e2 / sum;

	float expected_loss = -std::log(p0);

	EXPECT_NEAR(loss->GetValue()[0], expected_loss, 0.001f);

	// -------------------------
	// Backward reference
	// -------------------------
	// Softmax + CE gradient simplification:
	// dL/dlogits = p - y

	EXPECT_NEAR(logits->GetGradient()[0], p0 - 1.0f, 1e-5f);
	EXPECT_NEAR(logits->GetGradient()[1], p1 - 0.0f, 1e-5f);
	EXPECT_NEAR(logits->GetGradient()[2], p2 - 0.0f, 1e-5f);
}



TEST(Examples, Kapathy_Example2_Vec)
{
	auto x = Node<Vector>::Create({ 2.0f, 0.0f },  "x");
	auto w = Node<Vector>::Create({ -3.0f, 1.0f }, "w");
	auto b = Node<Vector>::Create({ 6.881375f },   "b");

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
	auto x = Node<Vector>::Create({ 2.0f, 0.0f }, "x");
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



TEST(Examples, Kapathy_Example2_Layer2D)
{
	auto x = Node<Vector>::Create({ 2.0f, 0.0f }, "x");
	auto L = Layer2D<Vector>(2, 1, Activation::Tanh);
	L.GetWeight(0)->GetValue().SetValue()[0] = -3.0f;
	L.GetWeight(0)->GetValue().SetValue()[1] = 1.0f;
	L.GetBias(0)->GetValue().SetValue()[0] = 6.881375f;

	auto o = L.Forward(x);

	o->Backwards();

	EXPECT_NEAR(x->GetValue()[0], 2.0f, 1e-5f);
	EXPECT_NEAR(x->GetGradient()[0], -1.5f, 1e-5f);

	EXPECT_NEAR(x->GetValue()[1], 0.0f, 1e-5f);
	EXPECT_NEAR(x->GetGradient()[1], 0.5f, 1e-5f);

	EXPECT_NEAR(L.GetWeight(0)->GetValue()[0], -3.0f, 1e-5f);
	EXPECT_NEAR(L.GetWeight(0)->GetGradient()[0], 1.0f, 1e-5f);

	EXPECT_NEAR(L.GetWeight(0)->GetValue()[1], 1.0f, 1e-5f);
	EXPECT_NEAR(L.GetWeight(0)->GetGradient()[1], 0.0f, 1e-5f);

	EXPECT_NEAR(L.GetBias(0)->GetValue()[0], 6.8814f, 1e-4f);
	EXPECT_NEAR(L.GetBias(0)->GetGradient()[0], 0.5f, 1e-5f);

	EXPECT_NEAR(o->GetValue()[0], 0.7071f, 1e-5f);
	EXPECT_NEAR(o->GetGradient()[0], 1.0f, 1e-5f);
}



TEST(Autograd, DiamondGraphGradient)
{
	auto a = Node<Scalar>::Create(1.0f);

	auto b = a * Node<Scalar>::Create(2.0f);
	auto c = a * Node<Scalar>::Create(3.0f);
	auto d = b + c;

	d->Backwards();

	EXPECT_NEAR(a->GetGradient()[0], 5.0f, 1e-5f);
}



TEST(Autograd, SharedNodeMultiply)
{
	auto x = Node<Scalar>::Create(3.0f);

	auto y = x * x;

	y->Backwards();

	EXPECT_NEAR(x->GetGradient()[0], 6.0f, 1e-5f);
}



TEST(Autograd, VectorMultiplyGradient)
{
	auto a = Node<Vector>::Create({ 2.0f, 3.0f });
	auto b = Node<Vector>::Create({ 4.0f, 5.0f });

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
	auto x = Node<Vector>::Create({ 1.0f, 2.0f, 3.0f });

	auto y = x->Sum();

	y->Backwards();

	EXPECT_NEAR(x->GetGradient()[0], 1.0f, 1e-5f);
	EXPECT_NEAR(x->GetGradient()[1], 1.0f, 1e-5f);
	EXPECT_NEAR(x->GetGradient()[2], 1.0f, 1e-5f);
}



TEST(Autograd, NeuronGradient)
{
	auto input = Node<Vector>::Create({ 1.0f, 2.0f });
	auto weight = Node<Vector>::Create({ 3.0f, 4.0f });
	auto bias = Node<Vector>::Create({ 5.0f });

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
	auto x = Node<Scalar>::Create(0.5f);

	auto y = x->Tanh();

	y->Backwards();

	float expected = 1.0f - std::tanh(0.5f) * std::tanh(0.5f);

	EXPECT_NEAR(x->GetGradient()[0], expected, 1e-5f);
}



TEST(Autograd, ReLU_Vector_Gradient)
{
	// -------------------------
	// Input: mixed values
	// -------------------------
	auto x = Node<Vector>::Create({ -2.0f, -0.5f, 0.0f, 1.5f, 3.0f });

	// forward
	auto y = x->ReLU();

	// expected forward result
	std::vector<float> expected_forward = { 0.0f, 0.0f, 0.0f, 1.5f, 3.0f };

	for (size_t i = 0; i < y->GetValue().GetLength(); i++)
		EXPECT_NEAR(y->GetValue()[i], expected_forward[i], 1e-5f);

	// -------------------------
	// Backward pass
	// -------------------------
	y->Backwards();

	// expected gradient:
	// ReLU'(x) = 0 if x <= 0 else 1
	std::vector<float> expected_grad = { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f };

	for (size_t i = 0; i < x->GetGradient().GetLength(); i++)
		EXPECT_NEAR(x->GetGradient()[i], expected_grad[i], 1e-5f);
}



TEST(Autograd, ReLU_FiniteDifference)
{
	auto x = Node<Vector>::Create({ -1.3f, 0.7f, 2.2f });

	auto y = x->ReLU();

	y->Backwards();

	auto grad = x->GetGradient().GetValue();

	for (size_t i = 0; i < x->GetValue().GetLength(); i++)
	{
		float eps = 1e-4f;

		auto x_pos = Node<Vector>::Create(x->GetValue());
		auto x_neg = Node<Vector>::Create(x->GetValue());

		x_pos->GetValue().SetValue()[i] += eps;
		x_neg->GetValue().SetValue()[i] -= eps;

		auto y_pos = x_pos->ReLU();
		auto y_neg = x_neg->ReLU();

		float diff = (y_pos->GetValue()[i] - y_neg->GetValue()[i]) / (2 * eps);

		EXPECT_NEAR(grad[i], diff, 1e-2f);
	}
}



TEST(Training, GradientChangesAfterUpdate)
{
	auto x = Node<Vector>::Create({ 1.0f, 2.0f });
	auto target = Node<Vector>::Create({ 1.0f });

	Neuron2D n(2, Activation::Tanh);

	// deterministic params
	n.GetWeight()->value.m_pValues[0] =  0.5f;
	n.GetWeight()->value.m_pValues[1] = -0.3f;
	n.GetBias()->value.m_pValues[0]   =  0.1f;

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



TEST(Training, MNist)
{
	MNist mnist = MNist();

	mnist.PrintTrainImage(0);
	mnist.PrintTrainImage(1);
	mnist.PrintTrainImage(2);

	std::vector<NodePtr<Matrix>> xs;
	std::vector<NodePtr<Matrix>> labels;

	std::vector<NodePtr<Matrix>> val_xs;
	std::vector<NodePtr<Matrix>> val_labels;

	for (size_t i = 0; i < mnist.GetTrainingNumberOfImages(); i++)
	{
		xs.push_back(Node<Matrix>::Create(mnist.GetTrainingImageData(i)));

		int label = mnist.GetTrainingLabelData(i);
		//Convert to one hot encoding
		std::vector<float> one_hot(10, 0.0f);
		one_hot[label] = 1.0f;

		labels.push_back(Node<Matrix>::Create(one_hot));
	}

	for (size_t i = 0; i < mnist.GetValidationNumberOfImages(); i++)
	{
		val_xs.push_back(Node<Matrix>::Create(mnist.GetValidationImageData(i)));

		int label = mnist.GetValidationLabelData(i);
		//Convert to one hot encoding
		std::vector<float> one_hot(10, 0.0f);
		one_hot[label] = 1.0f;

		val_labels.push_back(Node<Matrix>::Create(one_hot));
	}


	auto tape = TapeRecorder<Matrix>();
	SGD<Matrix> sgd(0.01f);
	const int batch_size = 8;

	{
		auto x = Node<Matrix>::CreateWithSize(784, "input");
		auto label = Node<Matrix>::CreateWithSize(10, "label");

		Layer3D L1(784, 128, Activation::Tanh, InitType::Xavier);
		Layer3D L2(128,  10, Activation::Identity, InitType::Xavier);

		auto h1 = L1.Forward(x);
		auto pred = L2.Forward(h1);
		pred->SetLabel("output");

		auto loss = ((pred - label)->ElementwiseMul(pred - label))->Sum();
		loss->SetLabel("loss");

		tape.Compile(loss);
		sgd.SetTrainableParams(tape);
	}

	auto input = tape.SetValue("input");
	auto label = tape.SetValue("label");
	auto output = tape.SetValue("output");
	auto loss = tape.SetValue("loss");

	tape.PrintTape();


	// Forward pass
	auto calculate_acc = [&]() {
		int correct = 0;
		int val_correct = 0;

		for (size_t i = 0; i < xs.size(); i++)
		{
			*input = xs[i]->GetValue();

			tape.Forward();

			//Convert one hot to class index
			int pred_class = 0;
			float max_val = output->GetValue()[0];
			int target_class = 0;
			for (int c = 1; c < 10; c++)
			{
				float val = output->GetValue()[c];
				if (val > max_val)
				{
					max_val = val;
					pred_class = c;
				}
				if (labels[i]->GetValue()[c] == 1.0f)
					target_class = c;
			}

			if (pred_class == target_class) correct++;
		}

		for (size_t i = 0; i < val_xs.size(); i++)
		{
			*input = val_xs[i]->GetValue();

			tape.Forward();

			//Convert one hot to class index
			int pred_class = 0;
			float max_val = output->GetValue()[0];
			int target_class = 0;
			for (int c = 1; c < 10; c++)
			{
				float val = output->GetValue()[c];
				if (val > max_val)
				{
					max_val = val;
					pred_class = c;
				}
				if (val_labels[i]->GetValue()[c] == 1.0f)
					target_class = c;
			}

			if (pred_class == target_class) val_correct++;
		}

		return std::tuple{ (float)correct / xs.size(), (float)val_correct / val_xs.size() };
	};


	//Training loop

	float epoch_loss = 0.0f;

	for (size_t epoch = 0; epoch < 3; epoch++)
	{
		ShuffleDataset(xs, labels);


		epoch_loss = 0.0f;

		for (size_t i = 0; i < xs.size(); i += batch_size)
		{
			size_t end = std::min(i + batch_size, xs.size());

			size_t actual_batch_size = end - i;

			float batch_loss = 0.0f;

			tape.ZeroGradients();

			for (size_t j = i; j < end; j++)
			{
				*input = xs[j]->GetValue();
				*label = labels[j]->GetValue();

				tape.Forward();
				tape.Backward();

				epoch_loss += loss->GetValue()[0];
				batch_loss += loss->GetValue()[0];
			}

			sgd.Step();
		}

		epoch_loss /= xs.size();
	}

	auto [train_acc, val_acc] = calculate_acc();

	EXPECT_LE(epoch_loss, 0.25f);
	EXPECT_GE(train_acc,  0.90f);
	EXPECT_GE(val_acc,    0.90f);
}



TEST(Training, MNist_Tensor4D)
{
	MNist mnist = MNist();

	std::vector<std::vector<float>> xs_train;
	std::vector<float> labels_train;

	std::vector<std::vector<float>> xs_val;
	std::vector<float> labels_val;

	for (size_t i = 0; i < mnist.GetTrainingNumberOfImages(); i++)
	{
		xs_train.push_back(mnist.GetTrainingImageData(i));
		labels_train.push_back((float)mnist.GetTrainingLabelData(i));
	}

	for (size_t i = 0; i < mnist.GetValidationNumberOfImages(); i++)
	{
		xs_val.push_back(mnist.GetValidationImageData(i));
		labels_val.push_back((float)mnist.GetValidationLabelData(i));
	}


	auto tape = TapeRecorder<Tensor4D>();
	SGD<Tensor4D> sgd(0.03f, 0.7f);
	const int batch_size = 1;

	{
		auto x = Node<Tensor4D>::Create(Tensor4D({ batch_size, 1, 28 * 28, 1 }), "input");
		auto label = Node<Tensor4D>::Create(Tensor4D({ batch_size, 1, 1, 1 }), "label");

		Layer4D L1(784, 128, Activation::ReLU, InitType::Xavier);
		Layer4D L2(128, 10, Activation::Identity, InitType::Xavier);

		auto h1 = L1.Forward(x);
		auto pred = L2.Forward(h1);
		pred->SetLabel("output");

		//auto loss = (  (pred - label)->ElementwiseMul(pred - label)  )->Sum();
		auto loss = pred->Softmax_CrossEntropy(label);
		loss->SetLabel("loss");

		tape.Compile(loss);
		sgd.SetTrainableParams(tape);
		std::cout << "Number of parameters: " << sgd.GetNumberOfParameters() << std::endl;
	}

	auto input = tape.SetValue("input");
	auto label = tape.SetValue("label");
	auto output = tape.SetValue("output");
	auto loss = tape.SetValue("loss");

	tape.PrintTape();


	// Forward pass
	auto calculate_acc = [&]() {
		int correct = 0;
		int val_correct = 0;

		for (size_t i = 0; i < xs_train.size();)
		{
			size_t actual_batch_size = 0;

			while (actual_batch_size < batch_size && i + actual_batch_size < xs_train.size())
			{
				input->SetColumn(actual_batch_size, 0, 0, xs_train[i + actual_batch_size]);
				label->SetColumn(actual_batch_size, 0, 0, { labels_train[i + actual_batch_size] });
				actual_batch_size++;
			}

			tape.Forward();

			for (size_t j = 0; j < actual_batch_size; j++)
			{
				//Convert one hot to class index
				int pred_class = (int)output->ArgMax(j, 0);
				int target_class = (int)labels_train[i + j];

				if (pred_class == target_class) correct++;
			}

			i += actual_batch_size;
		}


		for (size_t i = 0; i < xs_val.size();)
		{
			size_t actual_batch_size = 0;

			while (actual_batch_size < batch_size && i + actual_batch_size < xs_val.size())
			{
				input->SetColumn(actual_batch_size, 0, 0, xs_val[i + actual_batch_size]);
				label->SetColumn(actual_batch_size, 0, 0, { labels_val[i + actual_batch_size] });
				actual_batch_size++;
			}

			tape.Forward();

			for (size_t j = 0; j < actual_batch_size; j++)
			{
				//Convert one hot to class index
				int pred_class = (int)output->ArgMax(j, 0);
				int target_class = (int)labels_val[i + j];

				if (pred_class == target_class) val_correct++;
			}

			i += actual_batch_size;
		}

		return std::tuple{ (float)correct * 100.0f / xs_train.size(), (float)val_correct * 100.0f / xs_val.size() };
	};


	//Training loop

	float epoch_loss = 0.0f;

	for (size_t epoch = 0; epoch < 1; epoch++)
	{
		ShuffleDataset(xs_train, labels_train);


		epoch_loss = 0.0f;

		for (size_t i = 0; i < xs_train.size(); i += batch_size)
		{
			size_t end = std::min(i + batch_size, xs_train.size());

			float batch_loss = 0.0f;

			size_t actual_batch_size = 0;

			for (size_t j = i; j < end; j++)
			{
				input->SetColumn(actual_batch_size, 0, 0, xs_train[j]);
				label->SetColumn(actual_batch_size, 0, 0, { labels_train[j] });
				actual_batch_size++;
			}

			tape.Forward();
			tape.ZeroGradients();
			tape.Backward();

			epoch_loss += loss->At(0, 0, 0, 0);
			batch_loss += loss->At(0, 0, 0, 0);

			sgd.Step();
		}

		epoch_loss /= xs_train.size();

		sgd.SetLearningRate(0.95f * sgd.GetLearningRate());
	}

	auto [train_acc, val_acc] = calculate_acc();

	EXPECT_LE(epoch_loss, 0.95f);
	EXPECT_GE(train_acc, 0.70f);
	EXPECT_GE(val_acc, 0.70f);
}



TEST(Training, MNist_Tensor4D_CNN)
{
	MNist mnist = MNist();

	mnist.PrintTrainImage(0);
	mnist.PrintTrainImage(1);
	mnist.PrintTrainImage(2);

	std::vector<Tensor4D> xs_train;
	std::vector<float> labels_train;

	std::vector<Tensor4D> xs_val;
	std::vector<float> labels_val;

	for (size_t i = 0; i < mnist.GetTrainingNumberOfImages(); i++)
	{
		xs_train.push_back(mnist.GetTrainImageTensor(i));
		labels_train.push_back((float)mnist.GetTrainingLabelData(i));
	}

	for (size_t i = 0; i < mnist.GetValidationNumberOfImages(); i++)
	{
		xs_val.push_back(Tensor4D({ 1, 1, 28 * 28, 1 }, mnist.GetValidationImageData(i)));
		labels_val.push_back((float)mnist.GetValidationLabelData(i));
	}


	auto tape = TapeRecorder<Tensor4D>();
	SGD<Tensor4D> sgd(0.001f, 0.3f);
	const int batch_size = 1;

	{
		//auto x = Node<Tensor4D>::Create(Tensor4D({ batch_size, 1, 28, 28 }), "input");
		auto x = Node<Tensor4D>::Create(Tensor4D({ batch_size, 1, 28, 28 }), "input");
		auto label = Node<Tensor4D>::Create(Tensor4D({ batch_size, 1, 1, 1 }), "label");

		Conv2D conv1(1, 8, 3, 1, 1, Activation::ReLU);
		Conv2D conv2(8, 16, 3, 1, 1, Activation::ReLU);
		Layer4D L1(3136, 128, Activation::ReLU, InitType::Xavier);
		Layer4D L2(128, 10, Activation::Identity, InitType::Xavier);

		auto out_conv1 = conv1.Conv(x);
		auto out_conv2 = conv2.Conv(out_conv1)->MaxPool2D(2, 2);
		auto flattened = out_conv2->Flatten();

		flattened->SetLabel("flattened");

		auto h1 = L1.Forward(flattened);
		auto pred = L2.Forward(h1);
		pred->SetLabel("output");

		auto loss = pred->Softmax_CrossEntropy(label);
		loss->SetLabel("loss");

		tape.Compile(loss);
		sgd.SetTrainableParams(tape);
		std::cout << "Number of parameters: " << sgd.GetNumberOfParameters() << std::endl;
	}

	auto conv_weights = tape.SetValue("conv_weights");
	auto dconv_weights = tape.GetGradient("conv_weights");
	auto input = tape.SetValue("input");
	auto dinput = tape.GetGradient("input");
	auto flattened = tape.SetValue("flattened");
	auto dflattened = tape.GetGradient("flattened");
	auto label = tape.SetValue("label");
	auto output = tape.SetValue("output");
	auto loss = tape.SetValue("loss");

	tape.PrintTape();

	size_t train_size = xs_train.size() / 1;


	// Forward pass
	auto calculate_acc = [&]() {
		int correct = 0;
		int val_correct = 0;

		for (size_t i = 0; i < train_size;)
		{
			size_t actual_batch_size = 0;

			while (actual_batch_size < batch_size && i + actual_batch_size < xs_train.size())
			{
				input->ViewBatch(actual_batch_size).CopyFrom(xs_train[i + actual_batch_size]);
				label->SetColumn(actual_batch_size, 0, 0, { labels_train[i + actual_batch_size] });
				actual_batch_size++;
			}

			tape.Forward();

			for (size_t j = 0; j < actual_batch_size; j++)
			{
				//Convert one hot to class index
				int pred_class = (int)output->ArgMax(j, 0);
				int target_class = (int)labels_train[i + j];

				if (pred_class == target_class) correct++;
			}

			i += actual_batch_size;
		}


		for (size_t i = 0; i < xs_val.size();)
		{
			size_t actual_batch_size = 0;

			while (actual_batch_size < batch_size && i + actual_batch_size < xs_val.size())
			{
				input->ViewBatch(actual_batch_size).CopyFrom(xs_val[i + actual_batch_size]);
				label->SetColumn(actual_batch_size, 0, 0, { labels_val[i + actual_batch_size] });
				actual_batch_size++;
			}

			tape.Forward();

			for (size_t j = 0; j < actual_batch_size; j++)
			{
				//Convert one hot to class index
				int pred_class = (int)output->ArgMax(j, 0);
				int target_class = (int)labels_val[i + j];

				if (pred_class == target_class) val_correct++;
			}

			i += actual_batch_size;
		}

		return std::tuple{ (float)correct * 100.0f / train_size, (float)val_correct * 100.0f / xs_val.size() };
	};


	//Training loop

	float epoch_loss = 0.0f;

	for (size_t epoch = 0; epoch < 1; epoch++)
	{
		ShuffleDataset(xs_train, labels_train);


		epoch_loss = 0.0f;

		for (size_t i = 0; i < train_size; i += batch_size)
		{
			size_t end = std::min(i + batch_size, xs_train.size());

			float batch_loss = 0.0f;

			size_t actual_batch_size = 0;

			for (size_t j = i; j < end; j++)
			{
				input->ViewBatch(actual_batch_size).CopyFrom(xs_train[i + actual_batch_size]);
				label->SetColumn(actual_batch_size, 0, 0, { labels_train[i + actual_batch_size] });
				actual_batch_size++;
			}

			if (actual_batch_size != batch_size)
				continue;

			tape.Forward();
			tape.ZeroGradients();
			tape.Backward();

			epoch_loss += loss->At(0, 0, 0, 0) * actual_batch_size;
			batch_loss += loss->At(0, 0, 0, 0) * actual_batch_size;

			if (i / batch_size % (train_size / 50) == 0)
				std::cout << "Batch " << i / batch_size + 1 << " of " << train_size / batch_size << " batch_loss " << batch_loss / batch_size << std::endl;

			sgd.Step();
		}

		epoch_loss /= train_size;

		std::cout << "batch_size: " << batch_size << std::endl;
		std::cout << "Epoch " << epoch + 1 << " Loss: " << epoch_loss << std::endl;

		sgd.SetLearningRate(0.95f * sgd.GetLearningRate());
	}

	auto [train_acc, val_acc] = calculate_acc();
	EXPECT_LE(epoch_loss, 0.30f);
	EXPECT_GE(train_acc, 0.90f);
	EXPECT_GE(val_acc, 0.90f);
}



TEST(Training, MNist_Tensor4D_Batched)
{
	MNist mnist = MNist();

	std::vector<std::vector<float>> xs_train;
	std::vector<float> labels_train;

	std::vector<std::vector<float>> xs_val;
	std::vector<float> labels_val;

	for (size_t i = 0; i < mnist.GetTrainingNumberOfImages(); i++)
	{
		//xs.push_back(Node<Tensor>::Create(Tensor({ 28*28, 1 }, mnist.GetTrainingImageData(i))));
		xs_train.push_back(mnist.GetTrainingImageData(i));
		labels_train.push_back((float)mnist.GetTrainingLabelData(i));
	}

	for (size_t i = 0; i < mnist.GetValidationNumberOfImages(); i++)
	{
		xs_val.push_back(mnist.GetValidationImageData(i));
		labels_val.push_back((float)mnist.GetValidationLabelData(i));
	}


	auto tape = TapeRecorder<Tensor4D>();
	SGD<Tensor4D> sgd(0.03f, 0.7f);
	const int batch_size = 10;

	{
		auto x = Node<Tensor4D>::Create(Tensor4D({ batch_size, 1, 28 * 28, 1 }), "input");
		auto label = Node<Tensor4D>::Create(Tensor4D({ batch_size, 1, 1, 1 }), "label");

		Layer4D L1(784, 128, Activation::ReLU, InitType::Xavier);
		Layer4D L2(128, 10, Activation::Identity, InitType::Xavier);

		auto h1 = L1.Forward(x);
		auto pred = L2.Forward(h1);
		pred->SetLabel("output");

		//auto loss = (  (pred - label)->ElementwiseMul(pred - label)  )->Sum();
		auto loss = pred->Softmax_CrossEntropy(label);
		loss->SetLabel("loss");

		tape.Compile(loss);
		sgd.SetTrainableParams(tape);
		std::cout << "Number of parameters: " << sgd.GetNumberOfParameters() << std::endl;
	}

	auto input = tape.SetValue("input");
	auto label = tape.SetValue("label");
	auto output = tape.SetValue("output");
	auto loss = tape.SetValue("loss");

	tape.PrintTape();


	// Forward pass
	auto calculate_acc = [&]() {
		int correct = 0;
		int val_correct = 0;

		for (size_t i = 0; i < xs_train.size();)
		{
			size_t actual_batch_size = 0;

			while (actual_batch_size < batch_size && i + actual_batch_size < xs_train.size())
			{
				input->SetColumn(actual_batch_size, 0, 0, xs_train[i + actual_batch_size]);
				label->SetColumn(actual_batch_size, 0, 0, { labels_train[i + actual_batch_size] });
				actual_batch_size++;
			}

			tape.Forward();

			for (size_t j = 0; j < actual_batch_size; j++)
			{
				//Convert one hot to class index
				int pred_class = (int)output->ArgMax(j, 0);
				int target_class = (int)labels_train[i + j];

				if (pred_class == target_class) correct++;
			}

			i += actual_batch_size;
		}


		for (size_t i = 0; i < xs_val.size();)
		{
			size_t actual_batch_size = 0;

			while (actual_batch_size < batch_size && i + actual_batch_size < xs_val.size())
			{
				input->SetColumn(actual_batch_size, 0, 0, xs_val[i + actual_batch_size]);
				label->SetColumn(actual_batch_size, 0, 0, { labels_val[i + actual_batch_size] });
				actual_batch_size++;
			}

			tape.Forward();

			for (size_t j = 0; j < actual_batch_size; j++)
			{
				//Convert one hot to class index
				int pred_class = (int)output->ArgMax(j, 0);
				int target_class = (int)labels_val[i + j];

				if (pred_class == target_class) val_correct++;
			}

			i += actual_batch_size;
		}

		return std::tuple{ (float)correct * 100.0f / xs_train.size(), (float)val_correct * 100.0f / xs_val.size() };
	};


	//Training loop

	float epoch_loss = 0.0f;

	for (size_t epoch = 0; epoch < 1; epoch++)
	{
		ShuffleDataset(xs_train, labels_train);


		epoch_loss = 0.0f;

		for (size_t i = 0; i < xs_train.size(); i += batch_size)
		{
			size_t end = std::min(i + batch_size, xs_train.size());

			float batch_loss = 0.0f;

			size_t actual_batch_size = 0;

			for (size_t j = i; j < end; j++)
			{
				input->SetColumn(actual_batch_size, 0, 0, xs_train[j]);
				label->SetColumn(actual_batch_size, 0, 0, { labels_train[j] });
				actual_batch_size++;
			}

			tape.Forward();
			tape.ZeroGradients();
			tape.Backward();

			epoch_loss += loss->At(0, 0, 0, 0) * actual_batch_size;
			batch_loss += loss->At(0, 0, 0, 0);

			sgd.Step();
		}

		epoch_loss /= xs_train.size();

		sgd.SetLearningRate(0.95f * sgd.GetLearningRate());
	}

	auto [train_acc, val_acc] = calculate_acc();

	EXPECT_LE(epoch_loss, 1.50f);
	EXPECT_GE(train_acc, 0.70f);
	EXPECT_GE(val_acc, 0.70f);
}



TEST(Training, MNist_Fashion)
{
	MNist mnist = MNist("mnist-fashion");

	std::vector<NodePtr<Matrix>> xs;
	std::vector<NodePtr<Matrix>> labels;

	std::vector<NodePtr<Matrix>> val_xs;
	std::vector<NodePtr<Matrix>> val_labels;

	for (size_t i = 0; i < mnist.GetTrainingNumberOfImages(); i++)
	{
		xs.push_back(Node<Matrix>::Create(mnist.GetTrainingImageData(i)));

		int label = mnist.GetTrainingLabelData(i);
		//Convert to one hot encoding
		std::vector<float> one_hot(10, 0.0f);
		one_hot[label] = 1.0f;

		labels.push_back(Node<Matrix>::Create(one_hot));
	}

	for (size_t i = 0; i < mnist.GetValidationNumberOfImages(); i++)
	{
		val_xs.push_back(Node<Matrix>::Create(mnist.GetValidationImageData(i)));

		int label = mnist.GetValidationLabelData(i);
		//Convert to one hot encoding
		std::vector<float> one_hot(10, 0.0f);
		one_hot[label] = 1.0f;

		val_labels.push_back(Node<Matrix>::Create(one_hot));
	}


	auto tape = TapeRecorder<Matrix>();
	SGD<Matrix> sgd(0.01f);
	const int batch_size = 8;

	{
		auto x = Node<Matrix>::CreateWithSize(784, "input");
		auto label = Node<Matrix>::CreateWithSize(10, "label");

		Layer3D L1(784, 128, Activation::ReLU,     InitType::Xavier);
		Layer3D L2(128,  10, Activation::Identity, InitType::Xavier);

		auto h1 = L1.Forward(x);
		auto pred = L2.Forward(h1);
		pred->SetLabel("output");

		auto loss = ((pred - label)->ElementwiseMul(pred - label))->Sum();
		loss->SetLabel("loss");

		tape.Compile(loss);
		sgd.SetTrainableParams(tape);
	}

	auto input = tape.SetValue("input");
	auto label = tape.SetValue("label");
	auto output = tape.SetValue("output");
	auto loss = tape.SetValue("loss");

	tape.PrintTape();


	// Forward pass
	auto calculate_acc = [&]() {
		int correct = 0;
		int val_correct = 0;

		for (size_t i = 0; i < xs.size(); i++)
		{
			*input = xs[i]->GetValue();

			tape.Forward();

			//Convert one hot to class index
			int pred_class = 0;
			float max_val = output->GetValue()[0];
			int target_class = 0;
			for (int c = 1; c < 10; c++)
			{
				float val = output->GetValue()[c];
				if (val > max_val)
				{
					max_val = val;
					pred_class = c;
				}
				if (labels[i]->GetValue()[c] == 1.0f)
					target_class = c;
			}

			if (pred_class == target_class) correct++;
		}

		for (size_t i = 0; i < val_xs.size(); i++)
		{
			*input = val_xs[i]->GetValue();

			tape.Forward();

			//Convert one hot to class index
			int pred_class = 0;
			float max_val = output->GetValue()[0];
			int target_class = 0;
			for (int c = 1; c < 10; c++)
			{
				float val = output->GetValue()[c];
				if (val > max_val)
				{
					max_val = val;
					pred_class = c;
				}
				if (val_labels[i]->GetValue()[c] == 1.0f)
					target_class = c;
			}

			if (pred_class == target_class) val_correct++;
		}

		return std::tuple{ (float)correct / xs.size(), (float)val_correct / val_xs.size() };
		};


	//Training loop

	float epoch_loss = 0.0f;

	for (size_t epoch = 0; epoch < 3; epoch++)
	{
		ShuffleDataset(xs, labels);


		epoch_loss = 0.0f;

		for (size_t i = 0; i < xs.size(); i += batch_size)
		{
			size_t end = std::min(i + batch_size, xs.size());

			size_t actual_batch_size = end - i;

			float batch_loss = 0.0f;

			tape.ZeroGradients();

			for (size_t j = i; j < end; j++)
			{
				*input = xs[j]->GetValue();
				*label = labels[j]->GetValue();

				tape.Forward();
				tape.Backward();

				epoch_loss += loss->GetValue()[0];
				batch_loss += loss->GetValue()[0];
			}

			sgd.Step();
		}

		epoch_loss /= xs.size();
	}

	auto [train_acc, val_acc] = calculate_acc();

	EXPECT_LE(epoch_loss, 0.25f);
	EXPECT_GE(train_acc, 0.85f);
	EXPECT_GE(val_acc, 0.80f);
}



TEST(Training, MNist_Fashion_SCE)
{
	MNist mnist = MNist("mnist-fashion");

	std::vector<NodePtr<Matrix>> xs;
	std::vector<NodePtr<Matrix>> labels;

	std::vector<NodePtr<Matrix>> val_xs;
	std::vector<NodePtr<Matrix>> val_labels;

	for (size_t i = 0; i < mnist.GetTrainingNumberOfImages(); i++)
	{
		xs.push_back(Node<Matrix>::Create(mnist.GetTrainingImageData(i)));

		int label = mnist.GetTrainingLabelData(i);
		//Convert to one hot encoding
		std::vector<float> one_hot(10, 0.0f);
		one_hot[label] = 1.0f;

		//labels.push_back(Node<Matrix>::Create(one_hot));
		labels.push_back(Node<Matrix>::Create({ (float)label }));
	}

	for (size_t i = 0; i < mnist.GetValidationNumberOfImages(); i++)
	{
		val_xs.push_back(Node<Matrix>::Create(mnist.GetValidationImageData(i)));

		int label = mnist.GetValidationLabelData(i);
		//Convert to one hot encoding
		std::vector<float> one_hot(10, 0.0f);
		one_hot[label] = 1.0f;

		//val_labels.push_back(Node<Matrix>::Create(one_hot));
		val_labels.push_back(Node<Matrix>::Create({ (float)label }));
	}


	auto tape = TapeRecorder<Matrix>();
	SGD<Matrix> sgd(0.004f, 0.7f);

	{
		auto x = Node<Matrix>::CreateWithSize(784, "input");
		auto label = Node<Matrix>::CreateWithSize(1, "label");

		Layer3D L1(784, 128, Activation::ReLU, InitType::Xavier);
		Layer3D L2(128, 10, Activation::Identity, InitType::Xavier);

		auto h1 = L1.Forward(x);
		auto pred = L2.Forward(h1);
		pred->SetLabel("output");

		//auto loss = (  (pred - label)->ElementwiseMul(pred - label)  )->Sum();
		auto loss = pred->Softmax_CrossEntropy(label);
		loss->SetLabel("loss");

		tape.Compile(loss);
		sgd.SetTrainableParams(tape);
		std::cout << "Number of parameters: " << sgd.GetNumberOfParameters() << std::endl;
	}

	auto input = tape.SetValue("input");
	auto label = tape.SetValue("label");
	auto output = tape.SetValue("output");
	auto loss = tape.SetValue("loss");

	tape.PrintTape();
	size_t batch_size = 1;


	// Forward pass
	auto calculate_acc = [&]() {
		int correct = 0;
		int val_correct = 0;

		for (size_t i = 0; i < xs.size(); i++)
		{
			*input = xs[i]->GetValue();

			tape.Forward();

			//Convert one hot to class index
			int pred_class = (int)output->ArgMax();
			int target_class = (int)labels[i]->GetValue()[0];

			if (pred_class == target_class) correct++;
		}

		for (size_t i = 0; i < val_xs.size(); i++)
		{
			*input = val_xs[i]->GetValue();

			tape.Forward();

			//Convert one hot to class index
			int pred_class = (int)output->ArgMax();
			int target_class = (int)val_labels[i]->GetValue()[0];

			if (pred_class == target_class) val_correct++;
		}

		return std::tuple{ (float)correct * 100.0f / xs.size(), (float)val_correct * 100.0f / val_xs.size() };
	};


	//Training loop

	float epoch_loss = 0.0f;

	for (size_t epoch = 0; epoch < 3; epoch++)
	{
		ShuffleDataset(xs, labels);


		epoch_loss = 0.0f;

		for (size_t i = 0; i < xs.size(); i += batch_size)
		{
			size_t end = std::min(i + batch_size, xs.size());

			size_t actual_batch_size = end - i;

			float batch_loss = 0.0f;

			tape.ZeroGradients();

			for (size_t j = i; j < end; j++)
			{
				*input = xs[j]->GetValue();
				*label = labels[j]->GetValue();

				tape.Forward();
				tape.Backward();

				epoch_loss += loss->GetValue()[0];
				batch_loss += loss->GetValue()[0];
			}

			sgd.Step();
		}

		epoch_loss /= xs.size();

		sgd.SetLearningRate(0.95f * sgd.GetLearningRate());
	}

	auto [train_acc, val_acc] = calculate_acc();

	EXPECT_LE(epoch_loss, 0.40f);
	EXPECT_GE(train_acc, 0.80f);
	EXPECT_GE(val_acc, 0.80f);
}



TEST(Node, ValueIsMutable)
{
	auto n = Node<Scalar>::Create(2.0f);

	auto before = n->GetValue()[0];

	n->GetValue() -= 1.0f;

	auto after = n->GetValue()[0];

	EXPECT_NE(before, after);
}



TEST(Node, GradientIsMutable)
{
	auto n = Node<Scalar>::Create(2.0f);

	auto before = n->GetGradient()[0];

	n->GetGradient() -= 1.0f;

	auto after = n->GetGradient()[0];

	EXPECT_NE(before, after);
}



TEST(Autograd, Pack_NoDoubleGradientPropagation)
{
	auto x1 = Node<Vector>::Create({ 1.0f });
	auto x2 = Node<Vector>::Create({ 2.0f });
	auto x3 = Node<Vector>::Create({ 3.0f });

	std::vector<NodePtr<Vector>> inputs = { x1, x2, x3 };

	auto packed = Node<Vector>::CreateWithSize(3);
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
	auto x = Node<Scalar>::Create(2.0f);

	// diamond graph with shared dependency
	auto a = x * Node<Scalar>::Create(3.0f);
	auto b = x * Node<Scalar>::Create(4.0f);

	auto y = a + b;

	y->Backwards();

	// mathematically:
	// y = x*3 + x*4 = x*(3+4) = 7x
	// dy/dx = 7

	EXPECT_NEAR(x->GetGradient()[0], 7.0f, 1e-5f);
}



TEST(Autograd, DetectMultipleBackpropPathsExplosion)
{
	auto x = Node<Scalar>::Create(1.0f);

	auto a = x * Node<Scalar>::Create(2.0f);
	auto b = x * Node<Scalar>::Create(3.0f);
	auto c = x * Node<Scalar>::Create(4.0f);

	auto y = a + b + c;

	y->Backwards();

	// correct:
	// dy/dx = 2 + 3 + 4 = 9

	EXPECT_NEAR(x->GetGradient()[0], 9.0f, 1e-5f);
}



TEST(Training, XOR_OverfitSingleEpochSanity)
{
	auto x = Node<Vector>::Create({ 0.0f, 1.0f });
	auto y_true = Node<Vector>::Create({ 1.0f });

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
	std::vector<NodePtr<Vector>> xs = {
		Node<Vector>::Create({ 0.0f, 0.0f }),
		Node<Vector>::Create({ 0.0f, 1.0f }),
		Node<Vector>::Create({ 1.0f, 0.0f }),
		Node<Vector>::Create({ 1.0f, 1.0f })
	};

	std::vector<NodePtr<Vector>> label = {
		Node<Vector>::Create({ 0.0f }),
		Node<Vector>::Create({ 1.0f }),
		Node<Vector>::Create({ 1.0f }),
		Node<Vector>::Create({ 0.0f })
	};

	Layer2D<Vector> L1(2, 3, Activation::Tanh);
	Layer2D<Vector> L2(3, 1, Activation::Tanh);

	float lr = 0.1f;
	float epoch_loss = 0.0f;

	for (size_t epoch = 0; epoch < 250; epoch++)
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
		EXPECT_NEAR(diff->GetValue()[0], 0.0f, 0.3f);
	}
}



TEST(Training, XOR_Layer3D)
{
	std::vector<NodePtr<Matrix>> xs = {
		Node<Matrix>::Create(Matrix({ {0.0f}, {0.0f} })),
		Node<Matrix>::Create(Matrix({ {0.0f}, {1.0f} })),
		Node<Matrix>::Create(Matrix({ {1.0f}, {0.0f} })),
		Node<Matrix>::Create(Matrix({ {1.0f}, {1.0f} })),
	};

	std::vector<NodePtr<Matrix>> label = {
		Node<Matrix>::Create({{ 0.0f }}),
		Node<Matrix>::Create({{ 1.0f }}),
		Node<Matrix>::Create({{ 1.0f }}),
		Node<Matrix>::Create({{ 0.0f }})
	};

	Layer3D L1(2, 3, Activation::Tanh);
	Layer3D L2(3, 1, Activation::Tanh);

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
			EXPECT_EQ(params.size(), 4);

			loss->Backwards();

			for (size_t i = 0; i < L1.GetOutputLength(); i++)
			{
				L1.GetWeights()->GetValue() -= lr * L1.GetWeights()->GetGradient();
				L1.GetBiases()->GetValue()  -= lr * L1.GetBiases()->GetGradient();
			}
			for (size_t i = 0; i < L2.GetOutputLength(); i++)
			{
				L2.GetWeights()->GetValue() -= lr * L2.GetWeights()->GetGradient();
				L2.GetBiases()->GetValue()  -= lr * L2.GetBiases()->GetGradient();
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
		EXPECT_NEAR(diff->GetValue()[0], 0.0f, 0.15f);
	}
}



TEST(Autograd, ScalarLossSanity)
{
	auto x = Node<Vector>::Create({ 0.5f });
	auto y = Node<Vector>::Create({ 1.0f });

	auto diff = x - y;
	auto loss = (diff * diff)->Sum();

	loss->Backwards();

	EXPECT_NEAR(loss->GetValue()[0], 0.25f, 1e-5f);
	EXPECT_NEAR(x->GetGradient()[0], -1.0f, 1e-5f);
}



TEST(SGD, XOR)
{
	std::vector<NodePtr<Vector>> xs = {
		Node<Vector>::Create({ 0.0f, 0.0f }),
		Node<Vector>::Create({ 0.0f, 1.0f }),
		Node<Vector>::Create({ 1.0f, 0.0f }),
		Node<Vector>::Create({ 1.0f, 1.0f })
	};

	std::vector<NodePtr<Vector>> label = {
		Node<Vector>::Create({ 0.0f }),
		Node<Vector>::Create({ 1.0f }),
		Node<Vector>::Create({ 1.0f }),
		Node<Vector>::Create({ 0.0f })
	};

	Layer2D<Vector> L1(2, 3, Activation::Tanh);
	Layer2D<Vector> L2(3, 1, Activation::Tanh);

	SGD<Vector> sgd(0.1f);

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



TEST(SGD, XOR_Batched)
{
	std::vector<NodePtr<Vector>> xs = {
		Node<Vector>::Create({ 0.0f, 0.0f }),
		Node<Vector>::Create({ 0.0f, 1.0f }),
		Node<Vector>::Create({ 1.0f, 0.0f }),
		Node<Vector>::Create({ 1.0f, 1.0f })
	};

	std::vector<NodePtr<Vector>> labels = {
		Node<Vector>::Create({ 0.0f }),
		Node<Vector>::Create({ 1.0f }),
		Node<Vector>::Create({ 1.0f }),
		Node<Vector>::Create({ 0.0f })
	};

	Layer2D<Vector> L1(2, 3, Activation::Tanh);
	Layer2D<Vector> L2(3, 1, Activation::Tanh);

	SGD<Vector> sgd(0.1f);

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

			NodePtr<Vector> batch_loss = nullptr;

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
	std::vector<NodePtr<Vector>> xs;
	std::vector<NodePtr<Vector>> labels;

	const int N = 20;

	for (int i = 0; i < N; i++)
	{
		float x = -3.1415f + i * (6.2830f / (N - 1)); // [-pi, pi]
		float y = std::sin(x);

		xs.push_back(Node<Vector>::Create({ x }));
		labels.push_back(Node<Vector>::Create({ y }));
	}

	// simple MLP: 1 → 8 → 8 → 1
	Layer2D<Vector> L1(1, 8, Activation::Tanh);
	Layer2D<Vector> L2(8, 8, Activation::Tanh);
	Layer2D<Vector> L3(8, 1, Activation::Tanh);

	SGD<Vector> sgd(0.02f);

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
	std::vector<NodePtr<Vector>> xs;
	std::vector<NodePtr<Vector>> labels;

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

			xs.push_back(Node<Vector>::Create({ x, y }));

			// one-hot-ish target for MSE
			if (class_id == 0)
				labels.push_back(Node<Vector>::Create({ 0.0f }));
			else
				labels.push_back(Node<Vector>::Create({ 1.0f }));
		}
	}

	// MLP: 2 -> 32 -> 32 -> 1
	Layer2D<Vector> L1( 2, 32, Activation::Tanh);
	Layer2D<Vector> L2(32, 32, Activation::Tanh);
	Layer2D<Vector> L3(32,  1, Activation::Tanh);

	SGD<Vector> sgd(0.01f);
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

	for (size_t epoch = 0; epoch < 1300; epoch++)
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



TEST(TapeRecorder, Kapathy_Example2)
{
	auto x1 = Node<Scalar>::Create(2.0f, "x1");
	auto x2 = Node<Scalar>::Create(0.0f, "x2");

	auto w1 = Node<Scalar>::Create(-3.0f, "w1");
	auto w2 = Node<Scalar>::Create(1.0f, "w2");

	auto b = Node<Scalar>::Create(6.881375f, "b");

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

	auto graph = TapeRecorder<Scalar>();
	graph.Compile(o);

	graph.Forward();

	graph.PrintTape();

	graph.ZeroGradients();
	graph.Backward();

	EXPECT_NEAR(graph.GetValue("x1")[0], 2.0f, 1e-5f);
	EXPECT_NEAR(graph.GetGradient("x1")[0], -1.5f, 1e-5f);

	EXPECT_NEAR(graph.GetValue("x2")[0], 0.0f, 1e-5f);
	EXPECT_NEAR(graph.GetGradient("x2")[0], 0.5f, 1e-5f);

	EXPECT_NEAR(graph.GetValue("w1")[0], -3.0f, 1e-5f);
	EXPECT_NEAR(graph.GetGradient("w1")[0], 1.0f, 1e-5f);

	EXPECT_NEAR(graph.GetValue("w2")[0], 1.0f, 1e-5f);
	EXPECT_NEAR(graph.GetGradient("w2")[0], 0.0f, 1e-5f);
	
	EXPECT_NEAR(graph.GetValue("x1w1")[0], -6.0f, 1e-5f);
	EXPECT_NEAR(graph.GetGradient("x1w1")[0], 0.5f, 1e-5f);

	EXPECT_NEAR(graph.GetValue("x2w2")[0], 0.0f, 1e-5f);
	EXPECT_NEAR(graph.GetGradient("x2w2")[0], 0.5f, 1e-5f);

	EXPECT_NEAR(graph.GetValue("x1w1x2w2")[0], -6.0f, 1e-5f);
	EXPECT_NEAR(graph.GetGradient("x1w1x2w2")[0], 0.5f, 1e-5f);

	EXPECT_NEAR(graph.GetValue("b")[0], 6.8814f, 1e-4f);
	EXPECT_NEAR(graph.GetGradient("b")[0], 0.5f, 1e-5f);

	EXPECT_NEAR(graph.GetValue("n")[0], 0.8814f, 1e-4f);
	EXPECT_NEAR(graph.GetGradient("n")[0], 0.5f, 1e-5f);

	EXPECT_NEAR(graph.GetValue("o")[0], 0.7071f, 1e-5f);
	EXPECT_NEAR(graph.GetGradient("o")[0], 1.0f, 1e-5f);
}



TEST(TapeRecorder, VectorOpsMatchNodeBackprop)
{
	auto x1 = Node<Vector>::Create({ 2.0f, -3.0f }, "x1");
	auto x2 = Node<Vector>::Create({ 4.0f,  5.0f }, "x2");
	auto b  = Node<Vector>::Create({ 1.5f }, "b");

	// -----------------------------
	// Build graph
	// -----------------------------

	auto mul = x1->ElementwiseMul(x2);
	mul->SetLabel("mul");

	auto sum = mul->Sum();
	sum->SetLabel("sum");

	auto add = sum->ElementwiseAdd(b);
	add->SetLabel("add");

	auto out = add->Tanh();
	out->SetLabel("out");

	// -----------------------------
	// Reference autograd
	// -----------------------------

	out->Backwards();

	auto ref_x1_grad = x1->GetGradient();
	auto ref_x2_grad = x2->GetGradient();
	auto ref_b_grad = b->GetGradient();

	auto ref_out = out->GetValue();

	// -----------------------------
	// Tape version
	// -----------------------------

	TapeRecorder<Vector> tape;
	tape.Compile(out);

	auto tx1 = tape.SetValue("x1");
	auto tx2 = tape.SetValue("x2");
	auto tb = tape.SetValue("b");

	*tx1 = x1->GetValue();
	*tx2 = x2->GetValue();
	*tb = b->GetValue();

	tape.ZeroGradients();
	tape.Forward();
	tape.Backward();

	auto tape_x1_grad = tape.GetGradient("x1");
	auto tape_x2_grad = tape.GetGradient("x2");
	auto tape_b_grad = tape.GetGradient("b");

	auto tape_out = tape.GetValue("out");

	// -----------------------------
	// Compare outputs
	// -----------------------------

	EXPECT_NEAR((*tape_out)[0], ref_out[0], 1e-5f);

	// -----------------------------
	// Compare gradients
	// -----------------------------

	EXPECT_NEAR((*tape_x1_grad)[0], ref_x1_grad[0], 1e-5f);
	EXPECT_NEAR((*tape_x1_grad)[1], ref_x1_grad[1], 1e-5f);

	EXPECT_NEAR((*tape_x2_grad)[0], ref_x2_grad[0], 1e-5f);
	EXPECT_NEAR((*tape_x2_grad)[1], ref_x2_grad[1], 1e-5f);

	EXPECT_NEAR((*tape_b_grad)[0], ref_b_grad[0], 1e-5f);
}



TEST(TapeRecorder, GradientAccumulationMatchesTwoBackwardPasses)
{
	auto x = Node<Vector>::Create({ 2.0f, 3.0f }, "x");
	auto w = Node<Vector>::Create({ 4.0f, 5.0f }, "w");
	w->SetAsTrainable(true);

	auto y = (x->ElementwiseMul(w))->Sum();
	y->SetLabel("y");

	TapeRecorder<Vector> tape;
	tape.Compile(y);

	auto tx = tape.SetValue("x");
	auto tw = tape.SetValue("w");

	*tx = { 2.0f, 3.0f };
	*tw = { 4.0f, 5.0f };

	tape.ZeroGradients();

	// first sample
	tape.Forward();
	tape.Backward();

	auto grad_after_first = *tape.GetGradient("w");

	// second sample SAME INPUT
	tape.Forward();
	tape.Backward();

	auto grad_after_second = *tape.GetGradient("w");

	// should double
	EXPECT_NEAR(grad_after_second[0], grad_after_first[0] * 2.0f, 1e-5f);

	EXPECT_NEAR(grad_after_second[1], grad_after_first[1] * 2.0f, 1e-5f);
}



TEST(TapeRecorder, ZeroGradientsActuallyZerosEverything)
{
	auto x = Node<Vector>::Create({ 1.0f, 2.0f }, "x");
	auto y = x->Sum();

	TapeRecorder<Vector> tape;
	tape.Compile(y);

	tape.ZeroGradients();

	tape.Forward();
	tape.Backward();

	auto grad_before = *tape.GetGradient("x");

	EXPECT_NEAR(grad_before[0], 1.0f, 1e-5f);
	EXPECT_NEAR(grad_before[1], 1.0f, 1e-5f);

	tape.ZeroGradients();

	auto grad_after = *tape.GetGradient("x");

	EXPECT_NEAR(grad_after[0], 0.0f, 1e-5f);
	EXPECT_NEAR(grad_after[1], 0.0f, 1e-5f);
}



TEST(TapeRecorder, BatchAccumulationEqualsManualGradientSum)
{
	TapeRecorder<Vector> tape;

	auto x = Node<Vector>::Create({ 0.0f, 0.0f }, "x");
	auto w = Node<Vector>::Create({ 1.0f, 2.0f }, "w");
	w->SetAsTrainable(true);

	auto y = (x->ElementwiseMul(w))->Sum();

	tape.Compile(y);

	auto tx = tape.SetValue("x");

	tape.ZeroGradients();

	// sample 1
	*tx = { 1.0f, 2.0f };
	tape.Forward();
	tape.Backward();

	// sample 2
	*tx = { 3.0f, 4.0f };
	tape.Forward();
	tape.Backward();

	auto gw = tape.GetGradient("w");

	// expected:
	// grad sample1 = [1,2]
	// grad sample2 = [3,4]
	// accumulated = [4,6]

	EXPECT_NEAR((*gw)[0], 4.0f, 1e-5f);
	EXPECT_NEAR((*gw)[1], 6.0f, 1e-5f);
}



TEST(TapeRecorder, XOR)
{
	std::vector<NodePtr<Vector>> xs = {
		Node<Vector>::Create({ 0.0f, 0.0f }),
		Node<Vector>::Create({ 0.0f, 1.0f }),
		Node<Vector>::Create({ 1.0f, 0.0f }),
		Node<Vector>::Create({ 1.0f, 1.0f })
	};

	std::vector<NodePtr<Vector>> labels = {
		Node<Vector>::Create({ 0.0f }),
		Node<Vector>::Create({ 1.0f }),
		Node<Vector>::Create({ 1.0f }),
		Node<Vector>::Create({ 0.0f })
	};

	Layer2D<Vector> L1(2, 3, Activation::Tanh);
	Layer2D<Vector> L2(3, 1, Activation::Tanh);

	SGD<Vector> sgd(0.1f);

	auto x_ = Node<Vector>::Create({ 0.0f, 0.0f }, "x");
	auto label_ = Node<Vector>::Create({ 0.0f }, "label");
	auto l1_out = L1.Forward(x_);
	auto out_ = L2.Forward(l1_out);

	auto diff = out_ - label_;
	auto loss_ = diff * diff;
	out_->SetLabel("output");
	loss_->SetLabel("loss");

	auto tape = TapeRecorder<Vector>();
	tape.Compile(loss_);

	sgd.SetTrainableParams(tape);

	auto input  = tape.SetValue("x");
	auto label  = tape.SetValue("label");
	auto output = tape.SetValue("output");
	auto loss   = tape.SetValue("loss");

	float lr = 0.1f;
	float epoch_loss = 0.0f;

	for (size_t epoch = 0; epoch < 350; epoch++)
	{
		epoch_loss = 0.0f;

		for (size_t i = 0; i < xs.size(); i++)
		{
			*input = xs[i]->GetValue();
			*label = labels[i]->GetValue();

			tape.Forward();

			tape.ZeroGradients();
			tape.Backward();

			sgd.Step();

			epoch_loss += loss->GetValue()[0];
		}
	}

	EXPECT_NEAR(epoch_loss, 0.0f, 0.1f);

	for (size_t i = 0; i < xs.size(); i++)
	{
		*input = xs[i]->GetValue();

		tape.Forward();

		auto diff = *output - labels[i]->GetValue();
		EXPECT_NEAR(diff.GetValue()[0], 0.0f, 0.1f);
	}
}



TEST(TapeRecorder, Tensor4D_FlattenBackward)
{
	TapeRecorder<Tensor4D> tape;

	// -----------------------------
	// Input Tensor (B, C, H, W)
	// -----------------------------
	auto x = Node<Tensor4D>::Create(
		Tensor4D({ 2, 1, 2, 2 },
		{
			1.0f, 2.0f,
			3.0f, 4.0f,

			5.0f, 6.0f,
			7.0f, 8.0f
		}),
		"x"
	);

	// Flatten operation
	auto flat = x->Flatten();
	flat->SetLabel("flat");

	// Loss: sum over flattened output
	auto loss = flat->Sum();
	loss->SetLabel("loss");

	// -----------------------------
	// Compile graph into tape
	// -----------------------------
	tape.Compile(loss);

	auto tx = tape.SetValue("x");

	tape.ZeroGradients();

	// -----------------------------
	// Forward pass
	// -----------------------------
	tape.Forward();

	// -----------------------------
	// Backward pass
	// -----------------------------
	tape.Backward();

	// -----------------------------
	// Forward sanity check
	// -----------------------------
	EXPECT_NEAR(loss->GetValue().At(0, 0), 36.0f, 1e-5f);

	// -----------------------------
	// Gradient check (flat)
	// -----------------------------
	auto gflat = flat->GetGradient();

	for (int i = 0; i < 8; i++)
		EXPECT_NEAR(gflat[i], 1.0f, 1e-5f);

	// -----------------------------
	// Gradient back to input
	// -----------------------------
	auto gx = x->GetGradient();

	EXPECT_EQ(gx.GetBatches(), 2);
	EXPECT_EQ(gx.GetRows(), 2);
	EXPECT_EQ(gx.GetColumns(), 2);

	// Batch 0
	EXPECT_NEAR(gx.At(0, 0, 0, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(gx.At(0, 0, 0, 1), 1.0f, 1e-5f);
	EXPECT_NEAR(gx.At(0, 0, 1, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(gx.At(0, 0, 1, 1), 1.0f, 1e-5f);

	// Batch 1
	EXPECT_NEAR(gx.At(1, 0, 0, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(gx.At(1, 0, 0, 1), 1.0f, 1e-5f);
	EXPECT_NEAR(gx.At(1, 0, 1, 0), 1.0f, 1e-5f);
	EXPECT_NEAR(gx.At(1, 0, 1, 1), 1.0f, 1e-5f);
}



TEST(TapeRecorder, XOR_Tensor4D)
{
	Tensor4D data({ 4, 1, 2, 1 });
	data.SetColumn(0, 0, 0, { 0,0 });
	data.SetColumn(1, 0, 0, { 0,1 });
	data.SetColumn(2, 0, 0, { 1,0 });
	data.SetColumn(3, 0, 0, { 1,1 });

	Tensor4D data_labels({ 4, 1, 1, 1 }, { 0, 1, 1, 0 });


	NodePtr<Tensor4D> xs = Node<Tensor4D>::Create(data);
	NodePtr<Tensor4D> labels = Node<Tensor4D>::Create(data_labels);

	Layer4D L1(2, 3, Activation::Tanh);
	Layer4D L2(3, 1, Activation::Tanh);

	SGD<Tensor4D> sgd(0.1f);

	auto x_ = Node<Tensor4D>::Create(Tensor4D({ 4, 1, 2, 1 }), "x");
	auto label_ = Node<Tensor4D>::Create(Tensor4D({ 4, 1, 1, 1 }), "label");
	auto l1_out = L1.Forward(x_);
	auto out_ = L2.Forward(l1_out);

	auto diff = out_ - label_;
	auto loss_ = diff->ElementwiseMul(diff)->Sum();
	out_->SetLabel("output");
	loss_->SetLabel("loss");

	auto tape = TapeRecorder<Tensor4D>();
	tape.Compile(loss_);

	sgd.SetTrainableParams(tape);

	auto input = tape.SetValue("x");
	auto label = tape.SetValue("label");
	auto output = tape.SetValue("output");
	auto loss = tape.SetValue("loss");

	float lr = 0.1f;
	float epoch_loss = 0.0f;

	*input = xs->GetValue();
	*label = labels->GetValue();

	for (size_t epoch = 0; epoch < 350; epoch++)
	{
		epoch_loss = 0.0f;

		tape.Forward();

		tape.ZeroGradients();
		tape.Backward();

		sgd.Step();

		epoch_loss += loss->Data()[0];
	}

	EXPECT_NEAR(epoch_loss, 0.0f, 0.1f);

	*input = xs->GetValue();
	tape.Forward();

	for (size_t i = 0; i < xs->GetValue().GetBatches(); i++)
		EXPECT_NEAR(output->Data()[i], labels->GetValue().Data()[i], 0.1f);
}



TEST(TapeRecorder, XOR_Tensor4D_Flatten)
{
	Tensor4D data({ 4, 1, 1, 2 });
	data.SetRow(0, 0, 0, { 0,0 });
	data.SetRow(1, 0, 0, { 0,1 });
	data.SetRow(2, 0, 0, { 1,0 });
	data.SetRow(3, 0, 0, { 1,1 });

	Tensor4D data_labels({ 4, 1, 1, 1 }, { 0, 1, 1, 0 });


	NodePtr<Tensor4D> xs = Node<Tensor4D>::Create(data);
	NodePtr<Tensor4D> labels = Node<Tensor4D>::Create(data_labels);

	Layer4D L1(2, 3, Activation::Tanh);
	Layer4D L2(3, 1, Activation::Tanh);

	SGD<Tensor4D> sgd(0.1f);

	auto x_ = Node<Tensor4D>::Create(Tensor4D({ 4, 1, 1, 2 }), "x");
	auto label_ = Node<Tensor4D>::Create(Tensor4D({ 4, 1, 1, 1 }), "label");
	auto l1_out = L1.Forward(x_->Flatten());
	auto out_ = L2.Forward(l1_out);

	auto diff = out_ - label_;
	auto loss_ = diff->ElementwiseMul(diff)->Sum();
	out_->SetLabel("output");
	loss_->SetLabel("loss");

	auto tape = TapeRecorder<Tensor4D>();
	tape.Compile(loss_);

	sgd.SetTrainableParams(tape);

	auto input = tape.SetValue("x");
	auto label = tape.SetValue("label");
	auto output = tape.SetValue("output");
	auto loss = tape.SetValue("loss");

	float lr = 0.1f;
	float epoch_loss = 0.0f;

	*input = xs->GetValue();
	*label = labels->GetValue();

	for (size_t epoch = 0; epoch < 350; epoch++)
	{
		epoch_loss = 0.0f;

		tape.Forward();

		tape.ZeroGradients();
		tape.Backward();

		sgd.Step();

		epoch_loss += loss->Data()[0];
	}

	EXPECT_NEAR(epoch_loss, 0.0f, 0.1f);

	*input = xs->GetValue();
	tape.Forward();

	for (size_t i = 0; i < xs->GetValue().GetBatches(); i++)
		EXPECT_NEAR(output->Data()[i], labels->GetValue().Data()[i], 0.1f);
}



TEST(TapeRecorder, XOR_Tensor)
{
	Tensor data({ 2, 4 });
	data.SetColumn(0, { 0,0 });
	data.SetColumn(1, { 0,1 });
	data.SetColumn(2, { 1,0 });
	data.SetColumn(3, { 1,1 });

	Tensor data_labels({ 4 }, { 0, 1, 1, 0 });


	NodePtr<Tensor> xs     = Node<Tensor>::Create(data);
	NodePtr<Tensor> labels = Node<Tensor>::Create(data_labels);

	LayerTensor L1(2, 3, Activation::Tanh);
	LayerTensor L2(3, 1, Activation::Tanh);

	SGD<Tensor> sgd(0.1f);

	auto x_ = Node<Tensor>::Create(Tensor({ 2, 4 }), "x");
	auto label_ = Node<Tensor>::Create(Tensor({ 4 }), "label");
	auto l1_out = L1.Forward(x_);
	auto out_ = L2.Forward(l1_out);

	auto diff  = out_ - label_;
	auto loss_ = diff->ElementwiseMul(diff)->Sum();
	out_->SetLabel("output");
	loss_->SetLabel("loss");

	auto tape = TapeRecorder<Tensor>();
	tape.Compile(loss_);

	sgd.SetTrainableParams(tape);

	auto input  = tape.SetValue("x");
	auto label  = tape.SetValue("label");
	auto output = tape.SetValue("output");
	auto loss   = tape.SetValue("loss");

	float lr = 0.1f;
	float epoch_loss = 0.0f;

	*input = xs->GetValue();
	*label = labels->GetValue();

	for (size_t epoch = 0; epoch < 350; epoch++)
	{
		epoch_loss = 0.0f;

		tape.Forward();

		tape.ZeroGradients();
		tape.Backward();

		sgd.Step();

		epoch_loss += loss->Data()[0];
	}

	EXPECT_NEAR(epoch_loss, 0.0f, 0.2f);

	*input = xs->GetValue();
	tape.Forward();

	for (size_t i = 0; i < xs->GetValue().GetColumns(); i++)
		EXPECT_NEAR(output->Data()[i], labels->GetValue().Data()[i], 0.1f);
}



TEST(TapeRecorder, Conv2D_LearnsScalar)
{
	auto x = Node<Tensor4D>::Create(Tensor4D({ 1,1,1,1 }, { 2.0f }), "x");

	auto target = Node<Tensor4D>::Create(Tensor4D({ 1,1,1,1 }, { 10.0f }), "target");

	auto w = Node<Tensor4D>::Create(Tensor4D({ 1,1,1,1 }, { 1.0f }), "w");

	w->SetAsTrainable();

	auto y = x->Conv2D(w, 1, 0);

	auto diff = y - target;
	auto loss = diff * diff;

	TapeRecorder<Tensor4D> tape;
	tape.Compile(loss);

	SGD<Tensor4D> sgd(0.01f);
	sgd.SetTrainableParams(tape);

	for (int i = 0;i < 500;i++)
	{
		tape.Forward();

		tape.ZeroGradients();
		tape.Backward();

		sgd.Step();
	}

	EXPECT_NEAR(w->GetValue()[0], 5.0f, 0.1f);
	EXPECT_NEAR((*tape.GetValue("w"))[0], 5.0f, 0.1f);
}



TEST(TapeRecorder, ConvPoolLearnsSimplePattern)
{
	std::vector<NodePtr<Tensor4D>> xs =
	{
		Node<Tensor4D>::Create(
			Tensor4D({1,1,4,4},
			{
				1,0,0,0,
				0,0,0,0,
				0,0,0,0,
				0,0,0,0
			})),

		Node<Tensor4D>::Create(
			Tensor4D({1,1,4,4},
			{
				0,0,0,0,
				0,0,0,0,
				0,0,0,0,
				0,0,0,1
			}))
	};

	std::vector<NodePtr<Tensor4D>> ys =
	{
		Node<Tensor4D>::Create(
			Tensor4D({1,1,1,1},{0.0f})),

		Node<Tensor4D>::Create(
			Tensor4D({1,1,1,1},{1.0f}))
	};

	Conv2D conv(1, 1, 2, 1, 0, Activation::Tanh);

	Layer4D fc(4, 1, Activation::Identity);

	auto input = Node<Tensor4D>::Create(Tensor4D({ 1,1,4,4 }), "input");

	auto label = Node<Tensor4D>::Create(Tensor4D({ 1,1,1,1 }), "label");

	auto y = conv.Conv(input)->MaxPool2D(2, 2)->Flatten();

	auto out = fc.Forward(y);

	auto diff = out - label;
	auto loss = diff * diff;

	TapeRecorder<Tensor4D> tape;
	tape.Compile(loss);

	SGD<Tensor4D> sgd(0.05f);
	sgd.SetTrainableParams(tape);

	auto tInput = tape.SetValue("input");
	auto tLabel = tape.SetValue("label");

	for (int epoch = 0; epoch < 500; epoch++)
	{
		for (size_t i = 0; i < xs.size(); i++)
		{
			*tInput = xs[i]->GetValue();
			*tLabel = ys[i]->GetValue();

			tape.Forward();

			tape.ZeroGradients();
			tape.Backward();

			sgd.Step();
		}
	}

	// -----------------------
	// Evaluation
	// -----------------------

	*tInput = xs[0]->GetValue();
	tape.Forward();

	float pred0 = out->GetValue()[0];

	*tInput = xs[1]->GetValue();
	tape.Forward();

	float pred1 = out->GetValue()[0];

	EXPECT_LT(pred0, 0.2f);
	EXPECT_GT(pred1, 0.8f);
}



TEST(TapeRecorder, MaxPool2DGradientCheck)
{
	TapeRecorder<Tensor4D> tape;

	// Input: 1x1x4x4
	auto x = Node<Tensor4D>::Create(
		Tensor4D({ 1,1,4,4 },
		{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9,10,11,12,
			13,14,15,16
		}),
		"x"
	);

	// IMPORTANT: make x visible to tape
	tape.Compile(x);

	auto tx = tape.SetValue("x");

	tape.ZeroGradients();

	// Forward graph inside tape
	auto y = x->MaxPool2D(2, 2);
	auto loss = y->Sum();

	tape.Compile(loss);

	tape.Forward();
	tape.Backward();

	auto gx = x->GetGradient();

	// -------------------------
	// Expected gradient pattern:
	// -------------------------
	// Max per 2x2 block:
	//
	// Block1 -> 6
	// Block2 -> 8
	// Block3 -> 14
	// Block4 -> 16
	//
	// Gradient should be 1 at max positions only

	EXPECT_NEAR(gx.At(0, 0, 0, 0), 0.0f, 1e-5f);
	EXPECT_NEAR(gx.At(0, 0, 0, 1), 0.0f, 1e-5f);
	EXPECT_NEAR(gx.At(0, 0, 0, 2), 0.0f, 1e-5f);
	EXPECT_NEAR(gx.At(0, 0, 0, 3), 0.0f, 1e-5f);

	EXPECT_NEAR(gx.At(0, 0, 1, 0), 0.0f, 1e-5f);
	EXPECT_NEAR(gx.At(0, 0, 1, 1), 1.0f, 1e-5f);
	EXPECT_NEAR(gx.At(0, 0, 1, 2), 0.0f, 1e-5f);
	EXPECT_NEAR(gx.At(0, 0, 1, 3), 1.0f, 1e-5f);

	EXPECT_NEAR(gx.At(0, 0, 2, 0), 0.0f, 1e-5f);
	EXPECT_NEAR(gx.At(0, 0, 2, 1), 0.0f, 1e-5f);
	EXPECT_NEAR(gx.At(0, 0, 2, 2), 0.0f, 1e-5f);
	EXPECT_NEAR(gx.At(0, 0, 2, 3), 0.0f, 1e-5f);

	EXPECT_NEAR(gx.At(0, 0, 3, 0), 0.0f, 1e-5f);
	EXPECT_NEAR(gx.At(0, 0, 3, 1), 1.0f, 1e-5f);
	EXPECT_NEAR(gx.At(0, 0, 3, 2), 0.0f, 1e-5f);
	EXPECT_NEAR(gx.At(0, 0, 3, 3), 1.0f, 1e-5f);
}



TEST(TapeRecorder, MaxPool2DGradientCheck2)
{
	TapeRecorder<Tensor4D> tape;

	// Input: 1 sample, 1 channel, 2x2 kernel covers full 4x4 grid
	auto x = Node<Tensor4D>::Create(
		Tensor4D({ 1,1,4,4 },
		{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9,10,11,12,
			13,14,15,16
		}), "x");

		x->SetAsTrainable(true);

		// MaxPool 2x2 stride 2 => output 2x2
		auto y = x->MaxPool2D(2, 2);

		// Simple loss: sum of all outputs
		auto loss = y->Sum();

		tape.Compile(loss);

		tape.ZeroGradients();
		tape.Forward();
		tape.Backward();

		auto gx = tape.GetGradient("x");

		// Expected:
		// each output picks max of each 2x2 block:
		//
		// blocks:
		// [1 2; 5 6]   -> 6
		// [3 4; 7 8]   -> 8
		// [9 10; 13 14] -> 14
		// [11 12; 15 16] -> 16
		//
		// So gradient is:
		// only max positions get 1

		std::vector<float> expected = {
			0,0,0,0,
			0,1,0,1,
			0,0,0,0,
			0,1,0,1
		};

		for (int i = 0; i < 16; i++)
			EXPECT_NEAR(gx->Data()[i], expected[i], 1e-5f);
}



TEST(TapeRecorder, SpiralClassification_MSE)
{
	std::vector<NodePtr<Vector>> xs;
	std::vector<NodePtr<Vector>> labels;

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

			xs.push_back(Node<Vector>::Create({ x, y }));

			// one-hot-ish target for MSE
			if (class_id == 0)
				labels.push_back(Node<Vector>::Create({ 0.0f }));
			else
				labels.push_back(Node<Vector>::Create({ 1.0f }));
		}
	}

	// MLP: 2 -> 32 -> 32 -> 1
	Layer2D<Vector> L1( 2, 32, Activation::Tanh);
	Layer2D<Vector> L2(32, 32, Activation::Tanh);
	Layer2D<Vector> L3(32, 1, Activation::Tanh);

	SGD<Vector> sgd(0.01f);

	auto x_ = Node<Vector>::Create({ 0.0f, 0.0f }, "input");
	auto label_ = Node<Vector>::Create({ 0.0f }, "label");

	auto h1 = L1.Forward(x_);
	auto h2 = L2.Forward(h1);
	auto out_ = L3.Forward(h2);

	auto loss_ = (out_ - label_) * (out_ - label_);
	out_->SetLabel("output");
	loss_->SetLabel("loss");

	auto tape = TapeRecorder<Vector>();
	tape.Compile(loss_);
	sgd.SetTrainableParams(tape);

	auto input  = tape.SetValue("input");
	auto label  = tape.SetValue("label");
	auto output = tape.SetValue("output");
	auto loss   = tape.SetValue("loss");

	tape.PrintTape();

	//Training

	float epoch_loss = 0.0f;

	for (size_t epoch = 0; epoch < 1250; epoch++)
	{
		ShuffleDataset(xs, labels);

		epoch_loss = 0.0f;

		for (size_t i = 0; i < xs.size(); i++)
		{
			*input = xs[i]->GetValue();
			*label = labels[i]->GetValue();

			tape.Forward();

			tape.ZeroGradients();
			tape.Backward();

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
		*input = xs[i]->GetValue();

		tape.Forward();

		float pred = output->GetValue()[0];
		float target = labels[i]->GetValue()[0];

		int predicted_class = pred   > 0.5f ? 1 : 0;
		int target_class    = target > 0.5f ? 1 : 0;

		if (predicted_class == target_class)
			correct++;
	}

	float accuracy = (float)correct / xs.size();

	std::cout << "Final Accuracy: " << accuracy << std::endl;

	EXPECT_GT(accuracy, 0.90f);
}



TEST(TapeRecorder, SpiralClassification_MSE_Batching)
{
	std::vector<NodePtr<Vector>> xs;
	std::vector<NodePtr<Vector>> labels;

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

			xs.push_back(Node<Vector>::Create({ x, y }));

			// one-hot-ish target for MSE
			if (class_id == 0)
				labels.push_back(Node<Vector>::Create({ 0.0f }));
			else
				labels.push_back(Node<Vector>::Create({ 1.0f }));
		}
	}

	// MLP: 2 -> 32 -> 32 -> 1
	Layer2D<Vector> L1(2,  32, Activation::Tanh);
	Layer2D<Vector> L2(32, 32, Activation::Tanh);
	Layer2D<Vector> L3(32,  1, Activation::Tanh);

	SGD<Vector> sgd(0.002f);

	auto x_ = Node<Vector>::Create({ 0.0f, 0.0f }, "input");
	auto label_ = Node<Vector>::Create({ 0.0f }, "label");

	auto h1 = L1.Forward(x_);
	auto h2 = L2.Forward(h1);
	auto out_ = L3.Forward(h2);

	auto loss_ = (out_ - label_) * (out_ - label_);
	out_->SetLabel("output");
	loss_->SetLabel("loss");

	auto tape = TapeRecorder<Vector>();
	tape.Compile(loss_);
	sgd.SetTrainableParams(tape);

	auto input  = tape.SetValue("input");
	auto label  = tape.SetValue("label");
	auto output = tape.SetValue("output");
	auto loss   = tape.SetValue("loss");

	tape.PrintTape();

	size_t batch_size = 1;

	//Training

	float epoch_loss = 0.0f;

	for (size_t epoch = 0; epoch < 1500; epoch++)
	{
		ShuffleDataset(xs, labels);

		epoch_loss = 0.0f;

		for (size_t i = 0; i < xs.size(); i += batch_size)
		{
			size_t end = std::min(i + batch_size, xs.size());

			size_t actual_batch_size = end - i;

			float batch_loss = 0.0f;

			tape.ZeroGradients();

			for (size_t j = i; j < end; j++)
			{
				*input = xs[j]->GetValue();
				*label = labels[j]->GetValue();

				tape.Forward();
				tape.Backward();

				epoch_loss += loss->GetValue()[0];
				batch_loss += loss->GetValue()[0];
			}

			sgd.Step();
		}

		epoch_loss /= xs.size();

		if (epoch % 100 == 0)
			std::cout << "Epoch " << epoch << " Loss: " << epoch_loss << std::endl;
	}

	// evaluate classification accuracy
	int correct = 0;

	for (size_t i = 0; i < xs.size(); i++)
	{
		*input = xs[i]->GetValue();

		tape.Forward();

		float pred = output->GetValue()[0];
		float target = labels[i]->GetValue()[0];

		int predicted_class = pred   > 0.5f ? 1 : 0;
		int target_class    = target > 0.5f ? 1 : 0;

		if (predicted_class == target_class)
			correct++;
	}

	float accuracy = (float)correct / xs.size();

	std::cout << "Final Accuracy: " << accuracy << std::endl;

	EXPECT_GT(accuracy, 0.80f);
}



TEST(TapeRecorder, SpiralClassification_MSE_Layer3D)
{
	std::vector<NodePtr<Matrix>> xs;
	std::vector<NodePtr<Matrix>> labels;

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

			xs.push_back(Node<Matrix>::Create(Matrix({ x, y }), "input"));

			// one-hot-ish target for MSE
			if (class_id == 0)
				labels.push_back(Node<Matrix>::Create(Matrix({ 0.0f }), "label"));
			else
				labels.push_back(Node<Matrix>::Create(Matrix({ 1.0f }), "label"));
		}
	}

	// MLP: 2 -> 32 -> 32 -> 1
	Layer3D L1( 2,  32, Activation::Tanh);
	Layer3D L2(32,  32, Activation::Tanh);
	Layer3D L3(32,   1, Activation::Tanh);

	SGD<Matrix> sgd(0.01f);

	auto x_ = Node<Matrix>::Create(Matrix({ {0.0f}, {0.0f} }), "input");
	auto label_ = Node<Matrix>::Create(Matrix({{ 0.0f }}), "label");

	auto h1 = L1.Forward(x_);
	auto h2 = L2.Forward(h1);
	auto out_ = L3.Forward(h2);

	auto loss_ = (out_ - label_) * (out_ - label_);
	out_->SetLabel("output");
	loss_->SetLabel("loss");

	auto tape = TapeRecorder<Matrix>();
	tape.Compile(loss_);
	sgd.SetTrainableParams(tape);

	auto input = tape.SetValue("input");
	auto label = tape.SetValue("label");
	auto output = tape.SetValue("output");
	auto loss = tape.SetValue("loss");

	tape.PrintTape();

	//Training

	float epoch_loss = 0.0f;

	for (size_t epoch = 0; epoch < 1250; epoch++)
	{
		ShuffleDataset(xs, labels);

		epoch_loss = 0.0f;

		for (size_t i = 0; i < xs.size(); i++)
		{
			*input = xs[i]->GetValue();
			*label = labels[i]->GetValue();

			tape.Forward();

			tape.ZeroGradients();
			tape.Backward();

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
		*input = xs[i]->GetValue();

		tape.Forward();

		float pred = output->GetValue()[0];
		float target = labels[i]->GetValue()[0];

		int predicted_class = pred   > 0.5f ? 1 : 0;
		int target_class    = target > 0.5f ? 1 : 0;

		if (predicted_class == target_class)
			correct++;
	}

	float accuracy = (float)correct / xs.size();

	std::cout << "Final Accuracy: " << accuracy << std::endl;

	EXPECT_GT(accuracy, 0.90f);
}



TEST(LayerTensor, BatchIndependence)
{
	constexpr int input_size  = 4;
	constexpr int output_size = 3;
	constexpr int batch_size  = 2;

	LayerTensor layer(input_size, output_size, Activation::Identity, InitType::Xavier);

	Tensor x({ input_size, batch_size });

	x.SetColumn(0, { 1.0f, 0.0f, 0.0f, 0.0f });
	x.SetColumn(1, { 0.0f, 1.0f, 0.0f, 0.0f });

	auto node = Node<Tensor>::Create(x);
	auto y = layer.Forward(node);

	auto y0 = y->GetValue().ViewColumn(0);
	auto y1 = y->GetValue().ViewColumn(1);

	bool identical = true;
	for (int i = 0; i < output_size; i++)
	{
		if (std::abs(y0.At(1, i) - y1.At(1, i)) > 1e-5f)
		{
			identical = false;
			break;
		}
	}

	EXPECT_FALSE(identical) << "Batch outputs are identical → batch dimension ignored!";
}



TEST(LayerTensor, BatchConsistency)
{
	constexpr int input_size = 4;
	constexpr int output_size = 3;

	LayerTensor layer(input_size, output_size, Activation::Identity, InitType::Xavier);

	Tensor x1({ input_size, 1 });
	Tensor x2({ input_size, 2 });

	std::vector<float> sample = { 1, 2, 3, 4 };

	x1.SetColumn(0, sample);

	x2.SetColumn(0, sample);
	x2.SetColumn(1, { 5, 6, 7, 8 });

	auto y1 = layer.Forward(Node<Tensor>::Create(x1));
	auto y2 = layer.Forward(Node<Tensor>::Create(x2));

	auto y1_col0 = y1->GetValue().ViewColumn(0);
	auto y2_col0 = y2->GetValue().ViewColumn(0);

	for (int i = 0; i < output_size; i++)
	{
		EXPECT_NEAR(y1_col0.At(0, i), y2_col0.At(0, i), 1e-5f)
			<< "Batch=1 vs Batch=2 mismatch for same input";
	}
}



TEST(LayerTensor, NoAliasingBetweenBatchColumns)
{
	constexpr int input_size = 4;
	constexpr int output_size = 3;
	constexpr int batch_size = 2;

	LayerTensor layer(input_size, output_size, Activation::Identity, InitType::Xavier);

	Tensor x({ input_size, batch_size });

	x.SetColumn(0, { 1, 0, 0, 0 });
	x.SetColumn(1, { 0, 1, 0, 0 });

	auto y = layer.Forward(Node<Tensor>::Create(x));

	auto y0 = y->GetValue().ViewColumn(0);
	auto y1 = y->GetValue().ViewColumn(1);

	// Force mutation to detect shared memory
	y0.At(1, 0) += 1.0f;

	EXPECT_NE(y0.At(1, 0), y1.At(1, 0)) << "Aliasing detected: both batch columns share memory!";
}



TEST(Layer4D, BatchIndependence)
{
	constexpr int input_size = 4;
	constexpr int output_size = 3;
	constexpr int batch_size = 2;

	Layer4D layer(input_size, output_size, Activation::Identity, InitType::Xavier);

	Tensor4D x({ input_size, batch_size });

	x.SetColumn(0, 0, 0, { 1.0f, 0.0f, 0.0f, 0.0f });
	x.SetColumn(0, 0, 1, { 0.0f, 1.0f, 0.0f, 0.0f });

	auto node = Node<Tensor4D>::Create(x);
	auto y = layer.Forward(node);

	auto y0 = y->GetValue().ViewColumn(0);
	auto y1 = y->GetValue().ViewColumn(1);

	bool identical = true;
	for (int i = 0; i < output_size; i++)
	{
		if (std::abs(y0.At(1, i) - y1.At(1, i)) > 1e-5f)
		{
			identical = false;
			break;
		}
	}

	EXPECT_FALSE(identical) << "Batch outputs are identical → batch dimension ignored!";
}



TEST(Layer4D, BatchConsistency)
{
	constexpr int input_size = 4;
	constexpr int output_size = 3;

	Layer4D layer(input_size, output_size, Activation::Identity, InitType::Xavier);

	Tensor4D x1({ input_size, 1 });
	Tensor4D x2({ input_size, 2 });

	std::vector<float> sample = { 1, 2, 3, 4 };

	x1.SetRow(0, 0, 0, sample);

	x2.SetRow(0, 0, 0, sample);
	x2.SetRow(0, 0, 1, { 5, 6, 7, 8 });

	auto y1 = layer.Forward(Node<Tensor4D>::Create(x1));
	auto y2 = layer.Forward(Node<Tensor4D>::Create(x2));

	auto y1_col0 = y1->GetValue().ViewRow(0, 0, 0);
	auto y2_col0 = y2->GetValue().ViewRow(0, 0, 1);

	for (int i = 0; i < output_size; i++)
	{
		EXPECT_NEAR(y1_col0.At(0, i), y2_col0.At(0, i), 1e-5f)
			<< "Batch=1 vs Batch=2 mismatch for same input";
	}
}



TEST(Layer4D, NoAliasingBetweenBatchColumns)
{
	constexpr int input_size = 4;
	constexpr int output_size = 3;
	constexpr int batch_size = 2;

	Layer4D layer(input_size, output_size, Activation::Identity, InitType::Xavier);

	Tensor4D x({ input_size, batch_size });

	x.SetColumn(0, 0, 0, { 1, 0, 0, 0 });
	x.SetColumn(0, 0, 1, { 0, 1, 0, 0 });

	auto y = layer.Forward(Node<Tensor4D>::Create(x));

	auto y0 = y->GetValue().ViewColumn(0);
	auto y1 = y->GetValue().ViewColumn(1);

	// Force mutation to detect shared memory
	y0.At(1, 0) += 1.0f;

	EXPECT_NE(y0.At(1, 0), y1.At(1, 0)) << "Aliasing detected: both batch columns share memory!";
}