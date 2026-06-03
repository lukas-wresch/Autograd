#include "gtest/gtest.h"
#include "../src/tensor.h"



TEST(Tensor, ConstructorInitializesToZero)
{
    Tensor t({ 2, 3 });

    EXPECT_EQ(t.Shape().size(), 2);
    EXPECT_EQ(t.Shape()[0], 2);
    EXPECT_EQ(t.Shape()[1], 3);

    EXPECT_EQ(t.Size(), 6);

    for (size_t i = 0; i < t.Size(); i++)
        EXPECT_FLOAT_EQ(t.Data()[i], 0.0f);
}



TEST(Tensor, ConstructorFromDataAndShape)
{
    std::vector<float> data = { 1, 2, 3, 4, 5, 6 };

    Tensor t({ 2, 3 }, data);

    EXPECT_EQ(t.Shape()[0], 2);
    EXPECT_EQ(t.Shape()[1], 3);
    EXPECT_EQ(t.Size(), 6);

    EXPECT_FLOAT_EQ(t.Data()[0], 1.0f);
    EXPECT_FLOAT_EQ(t.Data()[1], 2.0f);
    EXPECT_FLOAT_EQ(t.Data()[5], 6.0f);
}



TEST(Tensor, ShapeIsCorrect)
{
    Tensor t({ 4, 2, 3 });

    ASSERT_EQ(t.Shape().size(), 3);

    EXPECT_EQ(t.Shape()[0], 4);
    EXPECT_EQ(t.Shape()[1], 2);
    EXPECT_EQ(t.Shape()[2], 3);

    EXPECT_EQ(t.Size(), 24);
}



TEST(Tensor, ViewSharesMemory)
{
    Tensor t({ 2, 3 });

    float* original = t.Data();

    Tensor v = t.View({ 6 });

    EXPECT_EQ(v.Size(), 6);

    // same underlying memory
    EXPECT_EQ(v.Data(), original);

    v.Data()[0] = 42.0f;

    EXPECT_FLOAT_EQ(t.Data()[0], 42.0f);
}



TEST(Tensor, ViewDoesNotCopy)
{
    Tensor t({ 2, 2 });

    t.Data()[0] = 5.0f;

    Tensor v = t.View({ 4 });

    EXPECT_FLOAT_EQ(v.Data()[0], 5.0f);
}



TEST(Tensor, StridesAreCorrectForRowMajor)
{
    Tensor t({ 2, 3 });

    const auto& strides = t.Strides();

    // row-major expected:
    // [3, 1]
    EXPECT_EQ(strides.size(), 2);

    EXPECT_EQ(strides[0], 3);
    EXPECT_EQ(strides[1], 1);
}



TEST(Tensor, ViewChangesShapeOnly)
{
    Tensor t({ 2, 3 });

    Tensor v = t.View({ 3, 2 });

    EXPECT_EQ(v.Shape()[0], 3);
    EXPECT_EQ(v.Shape()[1], 2);

    EXPECT_EQ(v.Size(), 6);
    EXPECT_EQ(t.Size(), 6);
}



TEST(Tensor, ViewModifiesOriginal)
{
    Tensor t({ 2, 2 });

    Tensor v = t.View({ 4 });

    v.Data()[2] = 99.0f;

    EXPECT_FLOAT_EQ(t.Data()[2], 99.0f);
}



TEST(Tensor, ViewArgMax)
{
    // 2x3 Tensor:
    // [ 1, 5, 2
    //   3, 4, 0 ]

    std::vector<float> data = {
        1, 5, 2,
        3, 4, 0
    };

    Tensor t({ 2, 3 }, data);

	EXPECT_EQ(t.ViewRow(0).At({ 0, 0 }), 1.0f);
    EXPECT_EQ(t.ViewRow(0).At({ 0, 1 }), 5.0f);
    EXPECT_EQ(t.ViewRow(0).At({ 0, 2 }), 2.0f);

    EXPECT_EQ(t.ViewRow(0).ArgMax(), 1);
    // Row 1 view: [3, 4, 0]
    EXPECT_EQ(t.ViewRow(1).ArgMax(), 1);

    EXPECT_EQ(t.ViewColumn(0).ArgMax(), 1);
    EXPECT_EQ(t.ViewColumn(1).ArgMax(), 0);
    EXPECT_EQ(t.ViewColumn(2).ArgMax(), 0);
}



TEST(Tensor, MoveConstructorTransfersOwnership)
{
    Tensor a({ 2, 2 });

    a.Data()[0] = 7.0f;

    Tensor b = std::move(a);

    EXPECT_EQ(b.Size(), 4);
    EXPECT_FLOAT_EQ(b.Data()[0], 7.0f);

    // optional depending on your design
    EXPECT_EQ(a.Data(), nullptr);
}