#include "gtest/gtest.h"
#include "../src/matrix.h"



TEST(Matrix, ConstructorInitializesToZero)
{
    Matrix m(2, 3);

    EXPECT_EQ(m.GetRows(), 2);
    EXPECT_EQ(m.GetColumns(), 3);
    EXPECT_EQ(m.GetLength(), 6);

    for (size_t i = 0; i < m.GetLength(); i++)
        EXPECT_FLOAT_EQ(m[i], 0.0f);
}



TEST(Matrix, ConstructorFromList)
{
    Matrix m({
              { 1.0f, 2.0f, 3.0f },
              { 4.0f, 5.0f, 6.0f }
             });

    EXPECT_EQ(m.GetRows(),    2);
    EXPECT_EQ(m.GetColumns(), 3);
    EXPECT_EQ(m.GetLength(),  6);

    EXPECT_FLOAT_EQ(m[0], 1.0f);
    EXPECT_FLOAT_EQ(m[1], 2.0f);
}



TEST(Matrix, Transpose)
{
    Matrix m({
              { 1.0f, 2.0f, 3.0f },
              { 4.0f, 5.0f, 6.0f }
        });

    m = m.Transpose();

    EXPECT_EQ(m.GetRows(),    3);
    EXPECT_EQ(m.GetColumns(), 2);
    EXPECT_EQ(m.GetLength(), 6);

    EXPECT_FLOAT_EQ(m[0], 1.0f);
    EXPECT_FLOAT_EQ(m[1], 4.0f);
}



TEST(Matrix, SetZero)
{
    Matrix m(2, 2);

    m += 5.0f;
    m.SetZero();

    for (size_t i = 0; i < m.GetLength(); i++)
        EXPECT_FLOAT_EQ(m[i], 0.0f);
}



TEST(Matrix, SetOne)
{
    Matrix m(2, 2);

    m.SetOne();

    for (size_t i = 0; i < m.GetLength(); i++)
        EXPECT_FLOAT_EQ(m[i], 1.0f);
}



TEST(Matrix, CopyConstructorCreatesDeepCopy)
{
    Matrix a(2, 2);

    a.SetOne();

    Matrix b = a;

    b += 1.0f;

    EXPECT_FLOAT_EQ(a[0], 1.0f);
    EXPECT_FLOAT_EQ(b[0], 2.0f);
}



TEST(Matrix, MoveConstructorTransfersOwnership)
{
    Matrix a(2, 2);

    a.SetOne();

    Matrix b = std::move(a);

    EXPECT_EQ(b.GetRows(), 2);
    EXPECT_EQ(b.GetColumns(), 2);

    EXPECT_FLOAT_EQ(b[0], 1.0f);

    EXPECT_EQ(a.GetRows(), 0);
    EXPECT_EQ(a.GetColumns(), 0);
}



TEST(Matrix, Addition)
{
    Matrix a(2, 2);
    Matrix b(2, 2);

    a.SetOne();
    b.SetOne();

    Matrix c = a + b;

    for (size_t i = 0; i < c.GetLength(); i++)
        EXPECT_FLOAT_EQ(c[i], 2.0f);
}



TEST(Matrix, AdditionAssignment)
{
    Matrix a(2, 2);
    Matrix b(2, 2);

    a.SetOne();
    b.SetOne();

    a += b;

    for (size_t i = 0; i < a.GetLength(); i++)
        EXPECT_FLOAT_EQ(a[i], 2.0f);
}



TEST(Matrix, SubtractionAssignment)
{
    Matrix a(2, 2);
    Matrix b(2, 2);

    a.SetOne();
    b.SetOne();

    a -= b;

    for (size_t i = 0; i < a.GetLength(); i++)
        EXPECT_FLOAT_EQ(a[i], 0.0f);
}



TEST(Matrix, ScalarMultiplication)
{
    Matrix a(2, 2);

    a.SetOne();

    Matrix b = 5.0f * a;

    for (size_t i = 0; i < b.GetLength(); i++)
        EXPECT_FLOAT_EQ(b[i], 5.0f);
}



TEST(Matrix, ElementwiseMultiplication)
{
    Matrix a({
        { 1.0f, -2.0f,  3.0f },
        { 4.0f,  0.0f, -6.0f }
        });

    Matrix b({
        { 7.0f,  8.0f, -9.0f },
        {-1.0f,  2.0f,  3.0f }
        });

    Matrix c = a.ElementwiseMul(b);

    EXPECT_EQ(c.GetRows(),    2);
    EXPECT_EQ(c.GetColumns(), 3);
    EXPECT_EQ(c.GetLength(),  6);

    // Row 0
    EXPECT_FLOAT_EQ(c.At(0, 0), 7.0f);   //  1 *  7
    EXPECT_FLOAT_EQ(c.At(0, 1), -16.0f);  // -2 *  8
    EXPECT_FLOAT_EQ(c.At(0, 2), -27.0f);  //  3 * -9

    // Row 1
    EXPECT_FLOAT_EQ(c.At(1, 0), -4.0f);   //  4 * -1
    EXPECT_FLOAT_EQ(c.At(1, 1), 0.0f);   //  0 *  2
    EXPECT_FLOAT_EQ(c.At(1, 2), -18.0f);  // -6 *  3
}



TEST(Matrix, MatrixMultiplication)
{
    Matrix a(2, 3);
    Matrix b(3, 2);

    float* av = a.SetValue();
    float* bv = b.SetValue();

    // A =
    // [1 2 3]
    // [4 5 6]

    av[0] = 1; av[1] = 2; av[2] = 3;
    av[3] = 4; av[4] = 5; av[5] = 6;

    // B =
    // [ 7  8]
    // [ 9 10]
    // [11 12]

    bv[0] = 7;  bv[1] = 8;
    bv[2] = 9;  bv[3] = 10;
    bv[4] = 11; bv[5] = 12;

    Matrix c = a * b;

    EXPECT_EQ(c.GetRows(), 2);
    EXPECT_EQ(c.GetColumns(), 2);

    EXPECT_FLOAT_EQ(c[0], 58.0f);
    EXPECT_FLOAT_EQ(c[1], 64.0f);
    EXPECT_FLOAT_EQ(c[2], 139.0f);
    EXPECT_FLOAT_EQ(c[3], 154.0f);
}



TEST(Matrix, MatrixMultiplication2)
{
    Matrix a(2, 2);
    Matrix b(2, 2);

    float* av = a.SetValue();
    float* bv = b.SetValue();

    // A =
    // [1 2 3]
    // [4 5 6]

    av[0] = 1; av[1] = 0;
    av[2] = 0; av[3] = 1;

    // B =
    // [ 7  8]
    // [ 9 10]
    // [11 12]

    bv[0] = 4;  bv[1] = 2;
    bv[2] = 3;  bv[3] = 7;

    Matrix c = a * b;

    EXPECT_EQ(c.GetRows(), 2);
    EXPECT_EQ(c.GetColumns(), 2);

    EXPECT_FLOAT_EQ(c[0], 4.0f);
    EXPECT_FLOAT_EQ(c[1], 2.0f);
    EXPECT_FLOAT_EQ(c[2], 3.0f);
    EXPECT_FLOAT_EQ(c[3], 7.0f);
}



TEST(Matrix, MatrixVectorMultiplication)
{
    Matrix a(2, 2);
    Matrix b(2, 1);

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

    Matrix c = a * b;

    EXPECT_EQ(c.GetRows(), 2);
    EXPECT_EQ(c.GetColumns(), 1);

    EXPECT_FLOAT_EQ(c[0], 13.0f);
    EXPECT_FLOAT_EQ(c[1], -7.0f);
}



TEST(Matrix, ReLU)
{
    Matrix m(1, 4);

    float* v = m.SetValue();

    v[0] = -1.0f;
    v[1] = 0.0f;
    v[2] = 2.0f;
    v[3] = -5.0f;

    Matrix r = m.ReLU();

    EXPECT_FLOAT_EQ(r[0], 0.0f);
    EXPECT_FLOAT_EQ(r[1], 0.0f);
    EXPECT_FLOAT_EQ(r[2], 2.0f);
    EXPECT_FLOAT_EQ(r[3], 0.0f);
}



TEST(Matrix, Heaviside)
{
    Matrix m(1, 4);

    float* v = m.SetValue();

    v[0] = -1.0f;
    v[1] = 0.0f;
    v[2] = 3.0f;
    v[3] = 10.0f;

    Matrix h = m.Heaviside();

    EXPECT_FLOAT_EQ(h[0], 0.0f);
    EXPECT_FLOAT_EQ(h[1], 0.0f);
    EXPECT_FLOAT_EQ(h[2], 1.0f);
    EXPECT_FLOAT_EQ(h[3], 1.0f);
}



TEST(Matrix, OutOfRangeThrows)
{
    Matrix m(2, 2);

    EXPECT_THROW(
    {
        float x = m[10];
        (void)x;
    }, std::out_of_range);
}



TEST(Matrix, AdditionSizeMismatchThrows)
{
    Matrix a(2, 2);
    Matrix b(3, 3);

    EXPECT_THROW(
    {
        Matrix c = a + b;
    }, std::runtime_error);
}



TEST(Matrix, MultiplicationSizeMismatchThrows)
{
    Matrix a(2, 3);
    Matrix b(4, 2);

    EXPECT_THROW(
    {
        Matrix c = a * b;
    }, std::runtime_error);
}