package graph

import (
	"maps"
	"slices"

	tensors "github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type OutputProcessor[T Number] struct {
	arr   []T
	shape []int
}

func (op *OutputProcessor[T]) get1D() []T {
	// It is possible for a tensor to have a larger space than what is needed for the current data
	return slices.Clone(op.arr[:op.shape[0]])
}

func (op *OutputProcessor[T]) get2D() [][]T {
	result := make([][]T, op.shape[0])
	for i := range result {
		result[i] = make([]T, op.shape[1])
		for j := range result[i] {
			result[i][j] = op.arr[i*op.shape[1]+j]
		}
	}
	return result
}

func (g *Graph) getOutputs() []any {
	result := make([]any, len(g.outputs))
	for index, output := range g.outputs {
		tensor := g.kernel.Get(output)
		if tensor == nil {
			result[index] = nil
			continue
		}
		switch tensor.DType {
		case tensors.Float:
			op := OutputProcessor[float32]{
				arr:   tensor.FloatData,
				shape: tensor.Shape,
			}
			if len(tensor.Shape) == 1 {
				result[index] = op.get1D()
			} else {
				result[index] = op.get2D()
			}
		case tensors.Double:
			op := OutputProcessor[float64]{
				arr:   tensor.DoubleData,
				shape: tensor.Shape,
			}
			if len(tensor.Shape) == 1 {
				result[index] = op.get1D()
			} else {
				result[index] = op.get2D()
			}
		case tensors.Int32:
			op := OutputProcessor[int32]{
				arr:   tensor.Int32Data,
				shape: tensor.Shape,
			}
			if len(tensor.Shape) == 1 {
				result[index] = op.get1D()
			} else {
				result[index] = op.get2D()
			}
		case tensors.Int64:
			op := OutputProcessor[int64]{
				arr:   tensor.Int64Data,
				shape: tensor.Shape,
			}
			if len(tensor.Shape) == 1 {
				result[index] = op.get1D()
			} else {
				result[index] = op.get2D()
			}
		case tensors.String:
			if len(tensor.Shape) == 1 {
				stringArr := make([]string, tensor.Shape[0])
				for i := range stringArr {
					stringArr[i] = string(tensor.StringData[i])
				}
				result[index] = stringArr
			} else if len(tensor.Shape) == 2 {
				stringArr2D := make([][]string, tensor.Shape[0])
				for i := range stringArr2D {
					stringArr2D[i] = make([]string, tensor.Shape[1])
					for j := range stringArr2D[i] {
						stringArr2D[i][j] = string(tensor.StringData[i*tensor.Shape[1]+j])
					}
				}
				result[index] = stringArr2D
			}
		case tensors.IntMap:
			mapSlice := make([]map[int64]float32, tensor.Shape[0])
			for i := range mapSlice {
				mapSlice[i] = maps.Clone(tensor.IntMap[i])
			}
			result[index] = mapSlice
		case tensors.StringMap:
			mapSlice := make([]map[string]float32, tensor.Shape[0])
			for i := range mapSlice {
				mapSlice[i] = maps.Clone(tensor.StringMap[i])
			}
			result[index] = mapSlice
		}
	}
	return result
}
