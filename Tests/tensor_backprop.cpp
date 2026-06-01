#include "gtest/gtest.h"
#include "../src/tensor.h"
#include "../src/node.h"
#include "../src/tape_recorder.h"



TEST(MatrixVsTensor, Compare)
{
	Matrix m({ 2.0f, 0.0f });
	Tensor t = Matrix2Tensor(m);

	EXPECT_NEAR(m.At(0, 0), t.At(0, 0), 1e-5f);
	EXPECT_NEAR(m.At(1, 0), t.At(1, 0), 1e-5f);
}



TEST(MatrixVsTensor, SimpleTanhGraph)
{
	// x: 2x1
	auto x   = Node<Matrix>::Create({ 2.0f, 0.0f }, "x");
	auto x_t = Node<Tensor>::Create(Matrix2Tensor(x->GetValue()), "x");

	// w: 2x1
	auto w   = Node<Matrix>::Create({ -3.0f, 1.0f }, "w");
	auto w_t = Node<Tensor>::Create(Matrix2Tensor(w->GetValue()), "w");

	// b: 1x1 (broadcast oder scalar-matrix)
	auto b   = Node<Matrix>::Create({ 6.881375f }, "b");
	auto b_t = Node<Tensor>::Create(Matrix2Tensor(b->GetValue()), "b");

	// elementwise multiply
	auto xw = w->ElementwiseMul(x);
	xw->SetLabel("xw");

	auto xw_t = w_t->ElementwiseMul(x_t);
	xw_t->SetLabel("xw");

	// sum
	auto xw_sum = xw->Sum();
	auto n = xw_sum + b;

	auto xw_sum_t = xw_t->Sum();
	auto n_t = xw_sum_t + b_t;

	n->SetLabel("n");
	n_t->SetLabel("n");

	auto o = n->Tanh();
	o->SetLabel("o");

	auto o_t = n_t->Tanh();
	o_t->SetLabel("o");

	o->Backwards();

	TapeRecorder tape(o_t);
	tape.Forward();
	tape.ZeroGradients();
	tape.Backward();

	// forward checks
	EXPECT_NEAR(x->GetValue().At(0, 0), tape.GetValue("x")->At(0, 0), 1e-5f);
	EXPECT_NEAR(x->GetValue().At(1, 0), tape.GetValue("x")->At(1, 0), 1e-5f);

	EXPECT_NEAR(w->GetValue().At(0, 0), tape.GetValue("w")->At(0, 0), 1e-5f);
	EXPECT_NEAR(w->GetValue().At(1, 0), tape.GetValue("w")->At(1, 0), 1e-5f);

	EXPECT_NEAR(b->GetValue().At(0, 0), tape.GetValue("b")->At(0, 0), 1e-4f);

	// forward expected intermediate (elementwise product)
	EXPECT_NEAR(xw->GetValue().At(0, 0), tape.GetValue("xw")->At(0, 0), 1e-5f);
	EXPECT_NEAR(xw->GetValue().At(1, 0), tape.GetValue("xw")->At(1, 0), 1e-5f);

	EXPECT_NEAR(o->GetValue().At(0, 0),  tape.GetValue("o")->At(0, 0), 1e-4f);

	// gradients (elementwise)
	EXPECT_NEAR(x->GetGradient().At(0, 0), tape.GetGradient("x")->At(0, 0), 1e-5f);
	EXPECT_NEAR(x->GetGradient().At(1, 0), tape.GetGradient("x")->At(1, 0), 1e-5f);

	EXPECT_NEAR(w->GetGradient().At(0, 0), tape.GetGradient("w")->At(0, 0), 1e-5f);
	EXPECT_NEAR(w->GetGradient().At(1, 0), tape.GetGradient("w")->At(1, 0), 1e-5f);

	EXPECT_NEAR(b->GetGradient().At(0, 0), tape.GetGradient("b")->At(0, 0), 1e-5f);

	EXPECT_NEAR(n->GetGradient().At(0, 0), tape.GetGradient("n")->At(0, 0), 1e-5f);

	EXPECT_NEAR(o->GetGradient().At(0, 0), tape.GetGradient("o")->At(0, 0), 1e-5f);
}



TEST(MatrixVsTensor, MatrixMultiplicationGraph)
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
	auto bias = Node<Matrix>::Create(Matrix({ 1.0f }), "bias");

	auto A_t = Node<Tensor>::Create(Matrix2Tensor(A->GetValue()), "A");
	auto B_t = Node<Tensor>::Create(Matrix2Tensor(B->GetValue()), "B");
	auto bias_t = Node<Tensor>::Create(Matrix2Tensor(bias->GetValue()), "bias");

	// Matrix multiplication
	auto C = A * B;
	C->SetLabel("C");
	auto C_t = A_t * B_t;
	C_t->SetLabel("C");

	// Sum reduction
	auto S = C->Sum();
	S->SetLabel("S");
	auto S_t = C_t->Sum();
	S_t->SetLabel("S");

	// Final output
	auto Y = S + bias;
	Y->SetLabel("Y");
	auto Y_t = S_t + bias_t;
	Y_t->SetLabel("Y");

	// backward pass
	Y->Backwards();

	TapeRecorder tape(Y_t);
	tape.Forward();
	tape.ZeroGradients();
	tape.Backward();

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

	EXPECT_NEAR(C->GetValue().At(0, 0), tape.GetValue("C")->At(0, 0), 1e-5f);
	EXPECT_NEAR(C->GetValue().At(0, 1), tape.GetValue("C")->At(0, 1), 1e-5f);
	EXPECT_NEAR(C->GetValue().At(1, 0), tape.GetValue("C")->At(1, 0), 1e-5f);
	EXPECT_NEAR(C->GetValue().At(1, 1), tape.GetValue("C")->At(1, 1), 1e-5f);

	// Sum = 58 + 64 + 139 + 154 = 415
	EXPECT_NEAR(S->GetValue().At(0, 0), tape.GetValue("S")->At(0, 0), 1e-5f);

	// Y = 416
	EXPECT_NEAR(Y->GetValue().At(0, 0), tape.GetValue("Y")->At(0, 0), 1e-5f);

	// -------------------------------------------------
	// Gradient checks
	// -------------------------------------------------

	// dY/dY = 1
	EXPECT_NEAR(Y->GetGradient().At(0, 0), tape.GetGradient("Y")->At(0, 0), 1e-5f);

	// dY/dS = 1
	EXPECT_NEAR(S->GetGradient().At(0, 0), tape.GetGradient("S")->At(0, 0), 1e-5f);

	// dY/dbias = 1
	EXPECT_NEAR(bias->GetGradient().At(0, 0), tape.GetGradient("bias")->At(0, 0), 1e-5f);

	// Since S = sum(C),
	// dS/dC is a matrix full of ones:
	//
	// [1 1]
	// [1 1]

	EXPECT_NEAR(C->GetGradient().At(0, 0), tape.GetGradient("C")->At(0, 0), 1e-5f);
	EXPECT_NEAR(C->GetGradient().At(0, 1), tape.GetGradient("C")->At(0, 1), 1e-5f);
	EXPECT_NEAR(C->GetGradient().At(1, 0), tape.GetGradient("C")->At(1, 0), 1e-5f);
	EXPECT_NEAR(C->GetGradient().At(1, 1), tape.GetGradient("C")->At(1, 1), 1e-5f);

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

	EXPECT_NEAR(A->GetGradient().At(0, 0), tape.GetGradient("A")->At(0, 0), 1e-5f);
	EXPECT_NEAR(A->GetGradient().At(0, 1), tape.GetGradient("A")->At(0, 1), 1e-5f);
	EXPECT_NEAR(A->GetGradient().At(0, 2), tape.GetGradient("A")->At(0, 2), 1e-5f);

	EXPECT_NEAR(A->GetGradient().At(1, 0), tape.GetGradient("A")->At(1, 0), 1e-5f);
	EXPECT_NEAR(A->GetGradient().At(1, 1), tape.GetGradient("A")->At(1, 1), 1e-5f);
	EXPECT_NEAR(A->GetGradient().At(1, 2), tape.GetGradient("A")->At(1, 2), 1e-5f);

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

	EXPECT_NEAR(B->GetGradient().At(0, 0), tape.GetGradient("B")->At(0, 0), 1e-5f);
	EXPECT_NEAR(B->GetGradient().At(0, 1), tape.GetGradient("B")->At(0, 1), 1e-5f);

	EXPECT_NEAR(B->GetGradient().At(1, 0), tape.GetGradient("B")->At(1, 0), 1e-5f);
	EXPECT_NEAR(B->GetGradient().At(1, 1), tape.GetGradient("B")->At(1, 1), 1e-5f);

	EXPECT_NEAR(B->GetGradient().At(2, 0), tape.GetGradient("B")->At(2, 0), 1e-5f);
	EXPECT_NEAR(B->GetGradient().At(2, 1), tape.GetGradient("B")->At(2, 1), 1e-5f);
}



/*TEST(MatrixVsTensor, RowVectorColumnVectorOuterProduct)
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



TEST(MatrixVsTensor, LargerMatrixGraphWithoutActivation)
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

	EXPECT_EQ(xw->GetValue().GetRows(), 3);
	EXPECT_EQ(xw->GetValue().GetColumns(), 3);
	EXPECT_EQ(xw->GetGradient().GetRows(), 3);
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



TEST(MatrixVsTensor, SoftmaxGradient)
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



TEST(MatrixVsTensor, SoftmaxCrossEntropy)
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
}*/