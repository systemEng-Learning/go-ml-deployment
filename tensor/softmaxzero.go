package tensor

import (
    "errors"
    "math"
)

// Compute Softmax Zero for a 1D slice
func computeSoftmaxZero[T Float32_64](values []T) {
    // Find the maximum value in the slice
    var vMax T = values[0]
    for _, v := range values {
        if v > vMax {
            vMax = v
        }
    }

    // Precompute exp(-vMax) for small values
    expNegVMax := T(math.Exp(-float64(vMax)))

    // Compute the exponentials and sum
    var sum T
    for i := range values {
        v := values[i]
        if v > 0.0000001 || v < -0.0000001 {
            values[i] = T(math.Exp(float64(v - vMax)))
        } else {
            values[i] *= expNegVMax
        }
        sum += values[i]
    }

    // Handle edge case where sum is zero
    if sum == 0 {
        for i := range values {
            values[i] = 0.5
        }
    } else {
        // Normalize the values
        for i := range values {
            values[i] /= sum
        }
    }
}

// Apply Softmax Zero to a Tensor
func (t *Tensor) SoftmaxZeroInPlace() error {
    // Ensure the tensor is 2D and of a supported type
    if len(t.Shape) != 2 || (t.DType != Float && t.DType != Double) {
        return errors.New("unsupported tensor shape or data type")
    }

    // Apply Softmax Zero row by row
    switch t.DType {
    case Float:
        for i := 0; i < t.Shape[0]; i++ {
            start := i * t.Shape[1]
            end := start + t.Shape[1]
            computeSoftmaxZero(t.FloatData[start:end])
        }
    case Double:
        for i := 0; i < t.Shape[0]; i++ {
            start := i * t.Shape[1]
            end := start + t.Shape[1]
            computeSoftmaxZero(t.DoubleData[start:end])
        }
    }
    return nil
}