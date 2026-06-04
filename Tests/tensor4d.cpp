#include "gtest/gtest.h"
#include "../src/tensor4d.h"



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

    EXPECT_FLOAT_EQ(m[0], 1.0f);
    EXPECT_FLOAT_EQ(m[1], 2.0f);

    m = m.Transpose();

    EXPECT_EQ(m.GetRows(),    3);
    EXPECT_EQ(m.GetColumns(), 2);
    EXPECT_EQ(m.GetSize(),    6);

    EXPECT_FLOAT_EQ(m[0], 1.0f);
    EXPECT_FLOAT_EQ(m[1], 4.0f);
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



TEST(Tensor4D, MoveConstructorTransfersOwnership)
{
    Tensor4D a({ 2, 2 });

    a.SetOne();

    Tensor4D b = std::move(a);

    EXPECT_EQ(b.GetRows(), 2);
    EXPECT_EQ(b.GetColumns(), 2);

    EXPECT_FLOAT_EQ(b[0], 1.0f);

    EXPECT_EQ(a.GetRows(), 0);
    EXPECT_EQ(a.GetColumns(), 0);
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



/*TEST(Tensor4D, ElementwiseMultiplication)
{
    Tensor4D a({ 2, 3 }, 
        { 1.0f, -2.0f,  3.0f, 4.0f,  0.0f, -6.0f });

    Tensor4D b({ 2,3 },
        { 7.0f,  8.0f, -9.0f, -1.0f,  2.0f, 3.0f });

    Tensor4D c = a.ElementwiseMul(b);

    EXPECT_EQ(c.GetRows(),    2);
    EXPECT_EQ(c.GetColumns(), 3);
    EXPECT_EQ(c.GetSize(),  6);

    // Row 0
    EXPECT_FLOAT_EQ(c.At(0, 0), 7.0f);   //  1 *  7
    EXPECT_FLOAT_EQ(c.At(0, 1), -16.0f);  // -2 *  8
    EXPECT_FLOAT_EQ(c.At(0, 2), -27.0f);  //  3 * -9

    // Row 1
    EXPECT_FLOAT_EQ(c.At(1, 0), -4.0f);   //  4 * -1
    EXPECT_FLOAT_EQ(c.At(1, 1), 0.0f);   //  0 *  2
    EXPECT_FLOAT_EQ(c.At(1, 2), -18.0f);  // -6 *  3
}



TEST(Tensor4D, Tensor4DMultiplication)
{
    Tensor4D a(2, 3);
    Tensor4D b(3, 2);

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

    Tensor4D c = a * b;

    EXPECT_EQ(c.GetRows(), 2);
    EXPECT_EQ(c.GetColumns(), 2);

    EXPECT_FLOAT_EQ(c[0], 58.0f);
    EXPECT_FLOAT_EQ(c[1], 64.0f);
    EXPECT_FLOAT_EQ(c[2], 139.0f);
    EXPECT_FLOAT_EQ(c[3], 154.0f);
}



TEST(Tensor4D, Tensor4DMultiplication2)
{
    Tensor4D a(2, 2);
    Tensor4D b(2, 2);

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

    Tensor4D c = a * b;

    EXPECT_EQ(c.GetRows(), 2);
    EXPECT_EQ(c.GetColumns(), 2);

    EXPECT_FLOAT_EQ(c[0], 4.0f);
    EXPECT_FLOAT_EQ(c[1], 2.0f);
    EXPECT_FLOAT_EQ(c[2], 3.0f);
    EXPECT_FLOAT_EQ(c[3], 7.0f);
}



TEST(Tensor4D, Tensor4DMultiplication3)
{
    Tensor4D a(1, 3);
    Tensor4D b(1, 1);

    float* av = a.SetValue();
    float* bv = b.SetValue();

    av[0] =  1.0f;
    av[1] = -0.5f;
    av[2] =  2.0f;

    bv[0] = 2.0f;

    Tensor4D c = a * b;

    EXPECT_EQ(c.GetRows(),    1);
    EXPECT_EQ(c.GetColumns(), 3);

    EXPECT_FLOAT_EQ(c[0],  2.0f);
    EXPECT_FLOAT_EQ(c[1], -1.0f);
    EXPECT_FLOAT_EQ(c[2],  4.0f);
}



TEST(Tensor4D, Tensor4DVectorMultiplication)
{
    Tensor4D a(2, 2);
    Tensor4D b(2, 1);

    EXPECT_EQ(a.GetRows(),    2);
    EXPECT_EQ(a.GetColumns(), 2);
    EXPECT_EQ(b.GetRows(),    2);
    EXPECT_EQ(b.GetColumns(), 1);

    float* av = a.SetValue();
    float* bv = b.SetValue();

    av[0] = 1; av[1] = 0;
    av[2] = 0; av[3] = 1;

    bv[0] = 13;
    bv[1] = -7;

    Tensor4D c = a * b;

    EXPECT_EQ(c.GetRows(), 2);
    EXPECT_EQ(c.GetColumns(), 1);

    EXPECT_FLOAT_EQ(c[0], 13.0f);
    EXPECT_FLOAT_EQ(c[1], -7.0f);
}



TEST(Tensor4D, ReLU)
{
    Tensor4D m(1, 4);

    float* v = m.SetValue();

    v[0] = -1.0f;
    v[1] = 0.0f;
    v[2] = 2.0f;
    v[3] = -5.0f;

    Tensor4D r = m.ReLU();

    EXPECT_FLOAT_EQ(r[0], 0.0f);
    EXPECT_FLOAT_EQ(r[1], 0.0f);
    EXPECT_FLOAT_EQ(r[2], 2.0f);
    EXPECT_FLOAT_EQ(r[3], 0.0f);
}



TEST(Tensor4D, Heaviside)
{
    Tensor4D m(1, 4);

    float* v = m.SetValue();

    v[0] = -1.0f;
    v[1] = 0.0f;
    v[2] = 3.0f;
    v[3] = 10.0f;

    Tensor4D h = m.Heaviside();

    EXPECT_FLOAT_EQ(h[0], 0.0f);
    EXPECT_FLOAT_EQ(h[1], 0.0f);
    EXPECT_FLOAT_EQ(h[2], 1.0f);
    EXPECT_FLOAT_EQ(h[3], 1.0f);
}



TEST(Tensor4D, OutOfRangeThrows)
{
    Tensor4D m(2, 2);

    EXPECT_THROW(
    {
        float x = m[10];
        (void)x;
    }, std::out_of_range);
}



TEST(Tensor4D, AdditionSizeMismatchThrows)
{
    Tensor4D a(2, 2);
    Tensor4D b(3, 3);

    EXPECT_THROW(
    {
        Tensor4D c = a + b;
    }, std::runtime_error);
}



TEST(Tensor4D, MultiplicationSizeMismatchThrows)
{
    Tensor4D a(2, 3);
    Tensor4D b(4, 2);

    EXPECT_THROW(
    {
        Tensor4D c = a * b;
    }, std::runtime_error);
}



TEST(Tensor4D, SoftmaxBasic)
{
    Tensor4D m(3, 1);

    float* v = m.SetValue();

    v[0] = 1.0f;
    v[1] = 2.0f;
    v[2] = 3.0f;

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
    Tensor4D m(3, 1);

    float* v = m.SetValue();

    v[0] = 0.0f;
    v[1] = 1.0f;
    v[2] = 2.0f;

    Tensor4D s = m.Softmax();

    EXPECT_TRUE(s[2] > s[1]);
    EXPECT_TRUE(s[1] > s[0]);
}



TEST(Tensor4D, CrossEntropyBasic)
{
    Tensor4D pred({
        { 0.7f, 0.2f, 0.1f }
        });

    Tensor4D target({
        { 1.0f, 0.0f, 0.0f }
        });

    Tensor4D loss = pred.CrossEntropy(target);

    // expected: -log(0.7)
    float expected = -std::log(0.7f);

    EXPECT_NEAR(loss.At(0, 0), expected, 0.001f);
}



TEST(Tensor4D, CrossEntropyOneHot)
{
    Tensor4D pred({
        { 0.1f, 0.8f, 0.1f }
        });

    Tensor4D target({
        { 0.0f, 1.0f, 0.0f }
        });

    Tensor4D loss = pred.CrossEntropy(target);

    float expected = -std::log(0.8f);

    EXPECT_NEAR(loss.At(0, 0), expected, 0.001f);
}



TEST(Tensor4D, CrossEntropyZeroClamp)
{
    Tensor4D pred({
        { 0.0f, 1.0f, 0.0f }
        });

    Tensor4D target({
        { 1.0f, 0.0f, 0.0f }
        });

    Tensor4D loss = pred.CrossEntropy(target);

    // should not be -inf because of epsilon
    EXPECT_FALSE(std::isinf(loss.At(0, 0)));
    EXPECT_FALSE(std::isnan(loss.At(0, 0)));

    // should be large but finite
    EXPECT_GT(loss.At(0, 0), 0.0f);
}*/