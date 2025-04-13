package tensor

func createOutputTensor(firstDtype, secondDtype DataType, shape []int) *Tensor {
	result := &Tensor{Shape: shape}
	if firstDtype == secondDtype {
		result.DType = firstDtype
	} else {
		if (firstDtype == Float && secondDtype == Double) || (secondDtype == Float && firstDtype == Double) {
			// both float types, set to double
			result.DType = Double
		} else if (firstDtype == Int32 && secondDtype == Int64) || (secondDtype == Int32 && firstDtype == Int64) {
			// both int types, set to long
			result.DType = Int64
		} else if (firstDtype == Int32 && (secondDtype == Float || secondDtype == Double)) ||
			(secondDtype == Int32 && (firstDtype == Float || firstDtype == Double)) ||
			(firstDtype == Int64 && (secondDtype == Float || secondDtype == Double)) ||
			(secondDtype == Int64 && (firstDtype == Float || firstDtype == Double)) {
			// one is a float and the other is an integer, set to double
			result.DType = Double
		}
	}
	result.Alloc()
	return result
}
