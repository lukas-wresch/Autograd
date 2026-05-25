#include "gtest/gtest.h"
#include <random>
#include "../src/node.h"
#include "../src/neuron.h"
#include "../src/layer.h"
#include "../src/sgd.h"
#include "../src/tape_recorder.h"
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



TEST(ScalarTests, BasicArithmeticAndBackprop)
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



TEST(VectorTests, ElementwiseMultiplyAddBackprop)
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



TEST(Neuron2DTests, ForwardAndBackpropTanh)
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
	auto x = Node<Matrix>::Create({ {2.0f}, {0.0f} }, "x");

	// w: 2x1
	auto w = Node<Matrix>::Create({ {-3.0f}, {1.0f} }, "w");

	// b: 1x1 (broadcast oder scalar-matrix)
	auto b = Node<Matrix>::Create({ {6.881375f} }, "b");

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
	auto bias = Node<Matrix>::Create({
		{ 1.0f }
		}, "bias");

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



/*TEST(MatrixBackprop, SoftmaxCrossEntropyGradient)
{
	// input logits
	auto x = Node<Matrix>::Create({
		{ 1.0f, 2.0f, 3.0f }
		}, "x");

	// softmax
	auto s = x->Softmax();

	// target class = index 2
	auto y = Node<Matrix>::Create({
		{ 0.0f, 0.0f, 1.0f }
		}, "y");

	// cross entropy (manual)
	auto log_s = s->Log();
	auto mul = y->ElementwiseMul(log_s);
	auto loss = mul->Sum() * -1.0f;

	loss->Backwards();

	// expected gradient: s - y
	Matrix soft = s->GetValue();

	EXPECT_NEAR(x->GetGradient().At(0, 0), soft[0] - 0.0f, 1e-5f);
	EXPECT_NEAR(x->GetGradient().At(0, 1), soft[1] - 0.0f, 1e-5f);
	EXPECT_NEAR(x->GetGradient().At(0, 2), soft[2] - 1.0f, 1e-5f);
}*/



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
	SGD<Matrix> sgd(0.06f);
	const int batch_size = 8;

	{
		auto x = Node<Matrix>::CreateWithSize(784, "input");
		auto label = Node<Matrix>::CreateWithSize(10, "label");

		Layer3D L1(784, 128, Activation::Tanh);
		Layer3D L2(128, 10, Activation::Identity);

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

			sgd.Step(1.0f / actual_batch_size);
		}

		epoch_loss /= xs.size();
	}

	auto [train_acc, val_acc] = calculate_acc();

	EXPECT_LE(epoch_loss, 0.25f);
	EXPECT_GE(train_acc,  0.90f);
	EXPECT_GE(val_acc,    0.90f);
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



TEST(Training, XOR_Layer3D)
{
	std::vector<NodePtr<Matrix>> xs = {
		Node<Matrix>::Create({ {0.0f}, {0.0f} }),
		Node<Matrix>::Create({ {0.0f}, {1.0f} }),
		Node<Matrix>::Create({ {1.0f}, {0.0f} }),
		Node<Matrix>::Create({ {1.0f}, {1.0f} }),
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
		EXPECT_NEAR(diff->GetValue()[0], 0.0f, 0.1f);
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



TEST(SGD, XOR_BATCHED)
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

	for (size_t epoch = 0; epoch < 300; epoch++)
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
	Layer2D<Vector> L1(2, 32, Activation::Tanh);
	Layer2D<Vector> L2(32, 32, Activation::Tanh);
	Layer2D<Vector> L3(32, 1, Activation::Tanh);

	SGD<Vector> sgd(0.01f);
	const int batch_size = 8;

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

		int predicted_class = pred > 0.5f ? 1 : 0;
		int target_class = target > 0.5f ? 1 : 0;

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

	SGD<Vector> sgd(0.03f);
	int batch_size = 16;

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

			sgd.Step(1.0f / actual_batch_size);
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

			xs.push_back(Node<Matrix>::Create({ {x}, {y} }));

			// one-hot-ish target for MSE
			if (class_id == 0)
				labels.push_back(Node<Matrix>::Create({{ 0.0f }}));
			else
				labels.push_back(Node<Matrix>::Create({{ 1.0f }}));
		}
	}

	// MLP: 2 -> 32 -> 32 -> 1
	Layer3D L1( 2,  32, Activation::Tanh);
	Layer3D L2(32,  32, Activation::Tanh);
	Layer3D L3(32,   1, Activation::Tanh);

	SGD<Matrix> sgd(0.01f);
	const int batch_size = 8;

	auto x_ = Node<Matrix>::Create({ {0.0f}, {0.0f} }, "input");
	auto label_ = Node<Matrix>::Create({{ 0.0f }}, "label");

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

		int predicted_class = pred > 0.5f ? 1 : 0;
		int target_class = target > 0.5f ? 1 : 0;

		if (predicted_class == target_class)
			correct++;
	}

	float accuracy = (float)correct / xs.size();

	std::cout << "Final Accuracy: " << accuracy << std::endl;

	EXPECT_GT(accuracy, 0.90f);
}