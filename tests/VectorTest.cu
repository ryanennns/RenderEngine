#include <gtest/gtest.h>
#include "../include/Structs.h"
#include "../include/helpers.h"

const double TOLERANCE = 1e-6;

TEST(VectorTest, it_calculates_normal_vector)
{
    const Vector v{3.0, 4.0, 0.0};
    const Vector expected{0.6, 0.8, 0.0}; // Normalized vector
    Vector result = vectorNormal(v);

    EXPECT_NEAR(result.x, expected.x, TOLERANCE);
    EXPECT_NEAR(result.y, expected.y, TOLERANCE);
    EXPECT_NEAR(result.z, expected.z, TOLERANCE);
}

TEST(VectorTest, it_adds_vectors)
{
    Vector a{1.0, 2.0, 3.0};
    Vector b{4.0, 5.0, 6.0};
    Vector expected{5.0, 7.0, 9.0};

    Vector result = vectorAdd(a, b);

    EXPECT_EQ(result, expected);
}

TEST(VectorTest, it_subtracts_vectors)
{
    Vector a{5.0, 6.0, 7.0};
    Vector b{2.0, 3.0, 4.0};
    Vector expected{3.0, 3.0, 3.0};

    Vector result = vectorSubtract(a, b);

    EXPECT_EQ(result, expected);
}

TEST(VectorTest, it_multiplies_vectors)
{
    Vector v{1.0, 2.0, 3.0};
    double scalar = 2.0;
    Vector expected{2.0, 4.0, 6.0};

    Vector result = vectorMultiply(v, scalar);

    EXPECT_EQ(result, expected);
}

TEST(VectorTest, it_calculates_vector_cross_product)
{
    Vector a{1.0, 0.0, 0.0};
    Vector b{0.0, 1.0, 0.0};
    Vector expected{0.0, 0.0, 1.0};

    Vector result = vectorCrossProduct(a, b);

    EXPECT_EQ(result, expected);
}

TEST(VectorTest, it_calculates_vector_dot_product)
{
    Vector a{10.0, 10.0, 10.0};
    Vector b{1.0, 2.0, 3.0};
    double expected = 60.0;

    double result = vectorDotProduct(a, b);

    EXPECT_NEAR(result, expected, TOLERANCE);
}
