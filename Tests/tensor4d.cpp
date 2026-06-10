#include "gtest/gtest.h"
#include "../src/tensor4d.h"
#include "../src/node.h"



TEST(Tensor4D, ConstructorInitializesToZero)
{
    Tensor4D m({ 2, 3 });

    EXPECT_EQ(m.GetRows(), 2);
    EXPECT_EQ(m.GetColumns(), 3);
    EXPECT_EQ(m.GetSize(), 6);

    for (size_t i = 0; i < m.GetSize(); i++)
        EXPECT_FLOAT_EQ(m[i], 0.0f);
}



TEST(Tensor4D, ConstructorFromList)
{
	Tensor4D m({ 2,3 },
               { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f });

    EXPECT_EQ(m.GetRows(),    2);
    EXPECT_EQ(m.GetColumns(), 3);
    EXPECT_EQ(m.GetSize(),  6);

    EXPECT_FLOAT_EQ(m[0], 1.0f);
    EXPECT_FLOAT_EQ(m[1], 2.0f);
}



TEST(Tensor4D, Transpose)
{
    Tensor4D m({ 2, 3 },
              { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f });

    EXPECT_FLOAT_EQ(m.At(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(m.At(0, 1), 2.0f);

    m = m.Transpose();

    EXPECT_EQ(m.GetRows(),    3);
    EXPECT_EQ(m.GetColumns(), 2);
    EXPECT_EQ(m.GetSize(),    6);

    EXPECT_FLOAT_EQ(m.At(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(m.At(0, 1), 4.0f);
}



TEST(Tensor4D, SetZero)
{
    Tensor4D m({ 2, 2 });

    m.SetZero();

    for (size_t i = 0; i < m.GetSize(); i++)
        EXPECT_FLOAT_EQ(m[i], 0.0f);

    m += Tensor4D({ 1 }, { 5.0f });

    for (size_t i = 0; i < m.GetSize(); i++)
        EXPECT_FLOAT_EQ(m[i], 5.0f);
}



TEST(Tensor4D, SetOne)
{
    Tensor4D m({ 2, 2 });

    m.SetOne();

    EXPECT_FLOAT_EQ(m.At(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(m.At(0, 1), 1.0f);
    EXPECT_FLOAT_EQ(m.At(1, 0), 1.0f);
    EXPECT_FLOAT_EQ(m.At(1, 1), 1.0f);
}



TEST(Tensor4D, SetIdentity)
{
    Tensor4D m({ 2, 2 });

    m.SetIdentity();

    EXPECT_FLOAT_EQ(m.At(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(m.At(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(m.At(1, 0), 0.0f);
    EXPECT_FLOAT_EQ(m.At(1, 1), 1.0f);
}



TEST(Tensor4D, Addition)
{
    Tensor4D a({ 2, 2 });
    Tensor4D b({ 2, 2 });

    a.SetIdentity();
    b.SetIdentity();

    Tensor4D c = a + b;

    EXPECT_FLOAT_EQ(c.At(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(c.At(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(c.At(1, 0), 0.0f);
    EXPECT_FLOAT_EQ(c.At(1, 1), 2.0f);
}



TEST(Tensor4D, AdditionAssignment)
{
    Tensor4D a({ 2, 2 });
    Tensor4D b({ 2, 2 });

    a.SetIdentity();
    b.SetIdentity();

    a += b;

    EXPECT_FLOAT_EQ(a.At(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(a.At(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(a.At(1, 0), 0.0f);
    EXPECT_FLOAT_EQ(a.At(1, 1), 2.0f);

    Tensor4D c({ 2, 2 });
    c += Tensor4D({ 1 }, { 5.0f });

    EXPECT_FLOAT_EQ(c.At(0, 0), 5.0f);
    EXPECT_FLOAT_EQ(c.At(0, 1), 5.0f);
    EXPECT_FLOAT_EQ(c.At(1, 0), 5.0f);
    EXPECT_FLOAT_EQ(c.At(1, 1), 5.0f);
}



TEST(Tensor4D, Subtraction)
{
    Tensor4D a({ 2, 2 });
    Tensor4D b({ 2, 2 });

    a.SetIdentity();
    b.SetIdentity();

    Tensor4D c = a - b;

    EXPECT_FLOAT_EQ(c.At(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(c.At(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(c.At(1, 0), 0.0f);
    EXPECT_FLOAT_EQ(c.At(1, 1), 0.0f);

    Tensor4D d = a - Tensor4D({ 1 }, { 1.0f });

    EXPECT_FLOAT_EQ(d.At(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(d.At(0, 1), -1.0f);
    EXPECT_FLOAT_EQ(d.At(1, 0), -1.0f);
    EXPECT_FLOAT_EQ(d.At(1, 1), 0.0f);
}



TEST(Tensor4D, SubtractionAssignment)
{
    Tensor4D a({ 2, 2 });
    Tensor4D b({ 2, 2 });

    a.SetIdentity();
    b.SetIdentity();

    a -= b;

    for (size_t i = 0; i < a.GetSize(); i++)
        EXPECT_FLOAT_EQ(a[i], 0.0f);
}



TEST(Tensor4D, ScalarMultiplication)
{
    Tensor4D a({ 2, 2 });

    a.SetIdentity();

    Tensor4D b = Tensor4D({ 1 }, { 5.0f }) * a;

    EXPECT_FLOAT_EQ(b.At(0, 0), 5.0f);
    EXPECT_FLOAT_EQ(b.At(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(b.At(1, 0), 0.0f);
    EXPECT_FLOAT_EQ(b.At(1, 1), 5.0f);
}



TEST(Tensor4D, ElementwiseMultiplication)
{
    Tensor4D a({ 2, 3 }, 
        { 1.0f, -2.0f,  3.0f, 4.0f,  0.0f, -6.0f });

    Tensor4D b({ 2,3 },
        { 7.0f,  8.0f, -9.0f, -1.0f,  2.0f, 3.0f });

    Tensor4D c = a * b;

    EXPECT_EQ(c.GetRows(),    2);
    EXPECT_EQ(c.GetColumns(), 3);
    EXPECT_EQ(c.GetSize(),    6);

    // Row 0
    EXPECT_FLOAT_EQ(c.At(0, 0), 7.0f);   //  1 *  7
    EXPECT_FLOAT_EQ(c.At(0, 1), -16.0f);  // -2 *  8
    EXPECT_FLOAT_EQ(c.At(0, 2), -27.0f);  //  3 * -9

    // Row 1
    EXPECT_FLOAT_EQ(c.At(1, 0), -4.0f);   //  4 * -1
    EXPECT_FLOAT_EQ(c.At(1, 1), 0.0f);   //  0 *  2
    EXPECT_FLOAT_EQ(c.At(1, 2), -18.0f);  // -6 *  3
}



TEST(Tensor4D, MatrixMultiplication)
{
    Tensor4D a({ 2, 3 });
    Tensor4D b({ 3, 2 });

    // A =
    // [1 2 3]
    // [4 5 6]

    a[0] = 1; a[1] = 2; a[2] = 3;
    a[3] = 4; a[4] = 5; a[5] = 6;

    // B =
    // [ 7  8]
    // [ 9 10]
    // [11 12]

    b[0] = 7;  b[1] = 8;
    b[2] = 9;  b[3] = 10;
    b[4] = 11; b[5] = 12;

    Tensor4D c = a % b;

    EXPECT_EQ(c.GetRows(), 2);
    EXPECT_EQ(c.GetColumns(), 2);

    EXPECT_FLOAT_EQ(c[0], 58.0f);
    EXPECT_FLOAT_EQ(c[1], 64.0f);
    EXPECT_FLOAT_EQ(c[2], 139.0f);
    EXPECT_FLOAT_EQ(c[3], 154.0f);
}



TEST(Tensor4D, MatrixMultiplication2)
{
    Tensor4D a({ 2, 2 });
    Tensor4D b({ 2, 2 });

    // A =
    // [1 2 3]
    // [4 5 6]

    a[0] = 1; a[1] = 0;
    a[2] = 0; a[3] = 1;

    // B =
    // [ 7  8]
    // [ 9 10]
    // [11 12]

    b[0] = 4;  b[1] = 2;
    b[2] = 3;  b[3] = 7;

    Tensor4D c = a % b;

    EXPECT_EQ(c.GetRows(), 2);
    EXPECT_EQ(c.GetColumns(), 2);

    EXPECT_FLOAT_EQ(c[0], 4.0f);
    EXPECT_FLOAT_EQ(c[1], 2.0f);
    EXPECT_FLOAT_EQ(c[2], 3.0f);
    EXPECT_FLOAT_EQ(c[3], 7.0f);
}



TEST(Tensor4D, Tensor4DMultiplication3)
{
    Tensor4D a({ 1, 3 });
    Tensor4D b({ 1, 1 });

    a[0] =  1.0f;
    a[1] = -0.5f;
    a[2] =  2.0f;

    b[0] = 2.0f;

    Tensor4D c = a * b;

    EXPECT_EQ(c.GetRows(),    1);
    EXPECT_EQ(c.GetColumns(), 3);

    EXPECT_FLOAT_EQ(c[0],  2.0f);
    EXPECT_FLOAT_EQ(c[1], -1.0f);
    EXPECT_FLOAT_EQ(c[2],  4.0f);
}



TEST(Tensor4D, MatrixMultiplicationBatch)
{
    // Shape: [2, 2] interpretiert als Batch = 2 Matrizen à 2x2
    Tensor4D a({ 2, 2, 2 }); // 2 batches, 2x2
    Tensor4D b({ 2, 2, 2 });

    // Batch 0:
    // A0 = [1 2]
    //      [3 4]
    a[0] = 1; a[1] = 2;
    a[2] = 3; a[3] = 4;

    // B0 = [5 6]
    //      [7 8]
    b[0] = 5; b[1] = 6;
    b[2] = 7; b[3] = 8;

    // Batch 1:
    // A1 = [2 0]
    //      [1 2]
    a[4] = 2; a[5] = 0;
    a[6] = 1; a[7] = 2;

    // B1 = [1 2]
    //      [3 4]
    b[4] = 1; b[5] = 2;
    b[6] = 3; b[7] = 4;

    Tensor4D c = a % b;

    // -------------------------
    // Batch 0 expected:
    // [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
    // [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
    // -------------------------
    EXPECT_FLOAT_EQ(c[0], 19.0f);
    EXPECT_FLOAT_EQ(c[1], 22.0f);
    EXPECT_FLOAT_EQ(c[2], 43.0f);
    EXPECT_FLOAT_EQ(c[3], 50.0f);

    // -------------------------
    // Batch 1 expected:
    // [2*1 + 0*3, 2*2 + 0*4] = [2, 4]
    // [1*1 + 2*3, 1*2 + 2*4] = [7, 10]
    // -------------------------
    EXPECT_FLOAT_EQ(c[4], 2.0f);
    EXPECT_FLOAT_EQ(c[5], 4.0f);
    EXPECT_FLOAT_EQ(c[6], 7.0f);
    EXPECT_FLOAT_EQ(c[7], 10.0f);
}



TEST(Tensor4D, MatrixMultiplicationBatchBroadcast)
{
    // (1 x 1 x 1 x 2)
    Tensor4D a({ 1, 1, 1, 2 });

    a[0] = 2.0f;
    a[1] = 3.0f;

    // (4 x 1 x 2 x 1)
    Tensor4D b({ 4, 1, 2, 1 });

    // Batch 0 -> [1,0]
    b[0] = 1.0f;
    b[1] = 0.0f;

    // Batch 1 -> [0,1]
    b[2] = 0.0f;
    b[3] = 1.0f;

    // Batch 2 -> [1,1]
    b[4] = 1.0f;
    b[5] = 1.0f;

    // Batch 3 -> [2,3]
    b[6] = 2.0f;
    b[7] = 3.0f;

    Tensor4D c = a % b;

    EXPECT_EQ(c.GetBatches(), 4);
    EXPECT_EQ(c.GetDepth(), 1);
    EXPECT_EQ(c.GetRows(), 1);
    EXPECT_EQ(c.GetColumns(), 1);

    // [2,3]·[1,0]
    EXPECT_FLOAT_EQ(c.At(0, 0, 0, 0), 2.0f);

    // [2,3]·[0,1]
    EXPECT_FLOAT_EQ(c.At(1, 0, 0, 0), 3.0f);

    // [2,3]·[1,1]
    EXPECT_FLOAT_EQ(c.At(2, 0, 0, 0), 5.0f);

    // [2,3]·[2,3]
    EXPECT_FLOAT_EQ(c.At(3, 0, 0, 0), 13.0f);
}



TEST(Tensor4D, Tensor4DVectorMultiplication)
{
    Tensor4D a({ 2, 2 });
    Tensor4D b({ 2, 1 });

    EXPECT_EQ(a.GetRows(),    2);
    EXPECT_EQ(a.GetColumns(), 2);
    EXPECT_EQ(b.GetRows(),    2);
    EXPECT_EQ(b.GetColumns(), 1);

    a[0] = 1; a[1] = 0;
    a[2] = 0; a[3] = 1;

    b[0] = 13;
    b[1] = -7;

    Tensor4D c = a % b;

    EXPECT_EQ(c.GetRows(), 2);
    EXPECT_EQ(c.GetColumns(), 1);

    EXPECT_FLOAT_EQ(c[0], 13.0f);
    EXPECT_FLOAT_EQ(c[1], -7.0f);
}



TEST(Tensor4D, ReLU)
{
    Tensor4D m({ 1, 4 });

    m[0] = -1.0f;
    m[1] = 0.0f;
    m[2] = 2.0f;
    m[3] = -5.0f;

    Tensor4D r = m.ReLU();

    EXPECT_FLOAT_EQ(r[0], 0.0f);
    EXPECT_FLOAT_EQ(r[1], 0.0f);
    EXPECT_FLOAT_EQ(r[2], 2.0f);
    EXPECT_FLOAT_EQ(r[3], 0.0f);
}



TEST(Tensor4D, Heaviside)
{
    Tensor4D m({ 1, 4 });

    m[0] = -1.0f;
    m[1] = 0.0f;
    m[2] = 3.0f;
    m[3] = 10.0f;

    Tensor4D h = m.Heaviside();

    EXPECT_FLOAT_EQ(h[0], 0.0f);
    EXPECT_FLOAT_EQ(h[1], 0.0f);
    EXPECT_FLOAT_EQ(h[2], 1.0f);
    EXPECT_FLOAT_EQ(h[3], 1.0f);
}



TEST(Tensor4D, OutOfRangeThrows)
{
    Tensor4D m({ 2, 2 });

    EXPECT_THROW(
    {
        float x = m[10];
        (void)x;
    }, std::out_of_range);
}



TEST(Tensor4D, AdditionSizeMismatchThrows)
{
    Tensor4D a({ 2, 2 });
    Tensor4D b({ 3, 3 });

    EXPECT_THROW(
    {
        Tensor4D c = a + b;
    }, std::runtime_error);
}



TEST(Tensor4D, MultiplicationSizeMismatchThrows)
{
    Tensor4D a({ 2, 3 });
    Tensor4D b({ 4, 2 });

    EXPECT_THROW(
    {
        Tensor4D c = a * b;
    }, std::runtime_error);
}



TEST(Tensor4D, MatrixMultiplicationSizeMismatchThrows)
{
    Tensor4D a({ 2, 3 });
    Tensor4D b({ 4, 2 });

    EXPECT_THROW(
        {
            Tensor4D c = a % b;
        }, std::runtime_error);
}



TEST(Tensor4D, SoftmaxBasic)
{
    Tensor4D m({ 3, 1 });

    m[0] = 1.0f;
    m[1] = 2.0f;
    m[2] = 3.0f;

    Tensor4D s = m.Softmax();

    // Werte sollten zwischen 0 und 1 liegen
    EXPECT_GT(s[0], 0.0f);
    EXPECT_GT(s[1], 0.0f);
    EXPECT_GT(s[2], 0.0f);

    EXPECT_LT(s[0], 1.0f);
    EXPECT_LT(s[1], 1.0f);
    EXPECT_LT(s[2], 1.0f);

    // Summe muss 1 sein
    float sum = s[0] + s[1] + s[2];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}



TEST(Tensor4D, SoftmaxOrdering)
{
    Tensor4D m({ 3, 1 });

    m[0] = 0.0f;
    m[1] = 1.0f;
    m[2] = 2.0f;

    Tensor4D s = m.Softmax();

    EXPECT_TRUE(s[2] > s[1]);
    EXPECT_TRUE(s[1] > s[0]);
}



TEST(Tensor4D, SoftmaxBatchIndependence)
{
    // Shape: [batch, depth, rows, cols] = [2, 1, 1, 3]
    Tensor4D m({ 2, 3, 1 });

    // Batch 0: [1, 2, 3]
    m[0] = 1.0f;
    m[1] = 2.0f;
    m[2] = 3.0f;

    // Batch 1: [10, 11, 12]
    m[3] = 10.0f;
    m[4] = 11.0f;
    m[5] = 12.0f;

    Tensor4D s = m.Softmax();

    // --- Batch 0 ---
    float sum0 = s[0] + s[1] + s[2];

    EXPECT_GT(s[0], 0.0f);
    EXPECT_GT(s[1], 0.0f);
    EXPECT_GT(s[2], 0.0f);
    EXPECT_NEAR(sum0, 1.0f, 1e-5f);

    // Softmax ordering innerhalb Batch 0
    EXPECT_TRUE(s[2] > s[1]);
    EXPECT_TRUE(s[1] > s[0]);

    // --- Batch 1 ---
    float sum1 = s[3] + s[4] + s[5];

    EXPECT_GT(s[3], 0.0f);
    EXPECT_GT(s[4], 0.0f);
    EXPECT_GT(s[5], 0.0f);
    EXPECT_NEAR(sum1, 1.0f, 1e-5f);

    // Softmax ordering innerhalb Batch 1
    EXPECT_TRUE(s[5] > s[4]);
    EXPECT_TRUE(s[4] > s[3]);
}



TEST(Tensor4D, CrossEntropyBasic)
{
	Tensor4D pred({ 3, 1 }, { 0.7f, 0.2f, 0.1f });

	Tensor4D target({ 3, 1 }, { 1.0f, 0.0f, 0.0f });

    Tensor4D loss = pred.CrossEntropy(target);

    // expected: -log(0.7)
    float expected = -std::log(0.7f);

    EXPECT_NEAR(loss.At(0, 0), expected, 0.001f);
}



TEST(Tensor4D, CrossEntropyOneHot)
{
	Tensor4D pred({ 3,1 }, { 0.1f, 0.8f, 0.1f });

    Tensor4D target({ 1 },  { 1.0f });

    Tensor4D loss = pred.CrossEntropy(target);

    float expected = -std::log(0.8f);

    EXPECT_NEAR(loss.At(0, 0), expected, 0.001f);
}



TEST(Tensor4D, CrossEntropyZeroClamp)
{
	Tensor4D pred({ 3,1 }, { 0.0f, 1.0f, 0.0f });

    Tensor4D target({ 3, 1 }, { 1.0f, 0.0f, 0.0f });

    Tensor4D loss = pred.CrossEntropy(target);

    // should not be -inf because of epsilon
    EXPECT_FALSE(std::isinf(loss.At(0, 0)));
    EXPECT_FALSE(std::isnan(loss.At(0, 0)));

    // should be large but finite
    EXPECT_GT(loss.At(0, 0), 0.0f);
}



TEST(Tensor4D, Sum)
{
    Tensor4D m({ 2, 3 }, {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
        });

    auto node = Node<Tensor4D>::Create(m, "x");
    auto sum = node->Sum();

    // Erwartung: 1+2+3+4+5+6 = 21
    EXPECT_NEAR(sum->GetValue().At(0, 0), 21.0f, 1e-5f);
}



TEST(Tensor4DBackprop, SumBackward)
{
    Tensor4D m({ 2, 2 }, {
        1.0f, 2.0f,
        3.0f, 4.0f
        });

    auto x = Node<Tensor4D>::Create(m, "x");

    auto s = x->Sum();
    s->SetLabel("sum");

    s->Backwards();

    // Forward check
    EXPECT_NEAR(s->GetValue().At(0, 0), 10.0f, 1e-5f);

    // Backward check:
    // d(sum)/d(x_ij) = 1 für alle Elemente

    EXPECT_NEAR(x->GetGradient().At(0, 0), 1.0f, 1e-5f);
    EXPECT_NEAR(x->GetGradient().At(0, 1), 1.0f, 1e-5f);
    EXPECT_NEAR(x->GetGradient().At(1, 0), 1.0f, 1e-5f);
    EXPECT_NEAR(x->GetGradient().At(1, 1), 1.0f, 1e-5f);
}



TEST(Tensor4DBackprop, SimpleTanhGraph)
{
    // x: 2x1
    auto x = Node<Tensor4D>::Create(Tensor4D({ 2, 1 }, { 2.0f, 0.0f }), "x");

    // w: 2x1
    auto w = Node<Tensor4D>::Create(Tensor4D({ 2, 1 }, { -3.0f, 1.0f }), "w");

    // b: 1x1 (broadcast oder scalar-matrix)
    auto b = Node<Tensor4D>::Create(Tensor4D({ 1, 1 }, { 6.881375f }), "b");

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



TEST(Tensor4DBackprop, MatrixMultiplicationGraph)
{
    // A: 2x3
    auto A = Node<Tensor4D>::Create(Tensor4D({ 2, 3 }, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }), "A");

    // B: 3x2
    auto B = Node<Tensor4D>::Create(Tensor4D({ 3, 2 }, { 7.0f,  8.0f, 9.0f, 10.0f, 11.0f, 12.0f }), "B");

    // bias: scalar matrix
    auto bias = Node<Tensor4D>::Create(Tensor4D({ 1, 1 }, { 1.0f }), "bias");

    // Matrix multiplication
    auto C = A % B;
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



TEST(Tensor4DBackprop, LargerMatrixGraphWithoutActivation)
{
    // x: 3x3
    auto x = Node<Tensor4D>::Create(Tensor4D({ 3, 3 }, { 1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f }), "x");

    // w: 3x3
    auto w = Node<Tensor4D>::Create(Tensor4D({ 3, 3 }, { 0.5f, -1.0f,  2.0f, 1.5f,  0.0f, -0.5f, 2.0f, -2.0f,  1.0f }), "w");

    // b: scalar matrix
    auto b = Node<Tensor4D>::Create(Tensor4D({ 1, 1 }, { 10.0f }), "b");

    // elementwise multiplication
    auto xw = x->ElementwiseMul(w);
    xw->SetLabel("xw");

    EXPECT_NEAR(xw->GetValue().At(0, 0), 0.5f, 1e-5f);

    EXPECT_EQ(xw->GetValue().GetRows(), 3);
    EXPECT_EQ(xw->GetValue().GetColumns(), 3);
    EXPECT_EQ(xw->GetGradient().GetRows(), 3);
    EXPECT_EQ(xw->GetGradient().GetColumns(), 3);

    // reduction sum
    auto s = xw->Sum();
    s->SetLabel("s");

    EXPECT_NEAR(xw->GetValue().At(0, 0), 0.5f, 1e-5f);

    // final output
    auto y = s + b;
    y->SetLabel("y");

    EXPECT_NEAR(xw->GetValue().At(0, 0), 0.5f, 1e-5f);

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



TEST(Tensor4DBackprop, SoftmaxGradient)
{
    // input
    auto x = Node<Tensor4D>::Create(Tensor4D({ 3, 1 }, { 1.0f, 2.0f, 3.0f }), "x");

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
    float s1 = s->GetValue().At(1, 0);
    float s2 = s->GetValue().At(2, 0);

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
    EXPECT_NEAR(x->GetGradient().At(1, 0), 0.0f, 1e-5f);
    EXPECT_NEAR(x->GetGradient().At(2, 0), 0.0f, 1e-5f);
}



TEST(Tensor4DBackprop, SoftmaxCrossEntropy)
{
    // logits (unnormalized scores)
    auto logits = Node<Tensor4D>::Create(Tensor4D({ 3, 1 }, { 2.0f, 1.0f, 0.1f }), "logits");

    // ground truth: class 0
    auto target = Node<Tensor4D>::Create(Tensor4D({ 1, 1 }, { 0.0f }), "target");

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