package graph

import (
	"fmt"
	"slices"

	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)

func (g *Graph) setupFor1DFloat32Input(index int, data []float32) error {
	shape := slices.Clone(g.shapes[index])
	if len(shape) == 1 && shape[0] == -1 {
		shape[0] = len(data)
	} else if len(shape) == 1 && len(data)%shape[0] != 0 {
		return fmt.Errorf("data of length %d cannot fit expected input of length %d", len(data), shape[0])
	} else if len(shape) == 2 &&
		((shape[0] == -1 && len(data)%shape[1] != 0) || (shape[0] > -1 && len(data) != shape[0]*shape[1])) {
		return fmt.Errorf("data of length %d cannnot fit expected input of shape %v", len(data), shape)
	} else if len(shape) == 2 && shape[0] == -1 {
		shape[0] = len(data) / shape[1]
	}
	t, err := g.kernel.Output(g.inputs[index], shape, g.dtypes[index])
	if err != nil {
		return err
	}
	if t.DType == tensor.Double {
		for i := range data {
			t.DoubleData[i] = float64(data[i])
		}
	} else {
		copy(t.FloatData, data)
	}
	return nil
}

func (g *Graph) setupFor2DFloat32Input(index int, data [][]float32) error {
	shape := slices.Clone(g.shapes[index])
	m := len(data)
	n := len(data[0])
	if len(shape) == 1 {
		return fmt.Errorf("input should be 1D, got 2D")
	}
	if shape[0] == -1 && n == shape[1] {
		shape[0] = m
	} else if n != shape[1] || (shape[0] > -1 && shape[0] != m) {
		return fmt.Errorf("expected input of shape %v, got [%d, %d]", shape, m, n)
	}
	t, err := g.kernel.Output(g.inputs[index], shape, g.dtypes[index])
	if err != nil {
		return err
	}
	if t.DType == tensor.Double {
		for x := range m {
			for y := range n {
				t.DoubleData[x*n+y] = float64(data[x][y])
			}
		}
	} else {
		for x := range m {
			for y := range n {
				t.FloatData[x*n+y] = data[x][y]
			}
		}
	}
	return nil
}

func (g *Graph) Execute1DFloat32(input []float32) error {
	length := len(g.inputs)
	if length > 1 && length != len(input) {
		return fmt.Errorf("args count not equal, got %d, wanted %d", len(input), length)
	}
	for i := range g.inputs {
		if g.dtypes[i] != tensor.Float && g.dtypes[i] != tensor.Double {
			return fmt.Errorf("expected float or double inputs, got %d", g.dtypes[i])
		}
		var expectedLen = g.shapes[i][0]
		if len(g.shapes[i]) == 2 {
			expectedLen *= g.shapes[i][1]
		}
		if expectedLen < 0 {
			expectedLen *= -1
		}
		if length > 1 && expectedLen > 1 {
			return fmt.Errorf("input %d expects %d samples but it will get only a single sample", i, expectedLen)
		}
		if length > 1 {
			shape := g.shapes[i]
			shape[0] = 1
			t, err := g.kernel.Output(g.inputs[i], shape, g.dtypes[i])
			if err != nil {
				return err
			}
			if t.DType == tensor.Double {
				t.DoubleData[0] = float64(input[i])
			} else {
				t.FloatData[0] = input[i]
			}
		} else {
			err := g.setupFor1DFloat32Input(i, input)
			if err != nil {
				return err
			}
		}
	}
	g.RunNodes()
	return nil
}

func (g *Graph) Execute2DFloat32(input [][]float32) error {
	// setups the input tensor
	length := len(g.inputs)
	if length > 1 && length != len(input) {
		return fmt.Errorf("args count not equal, got %d, wanted %d", len(input), length)
	}
	for index := range g.inputs {
		if g.dtypes[index] != tensor.Float && g.dtypes[index] != tensor.Double {
			return fmt.Errorf("expected float or double inputs, got %d", g.dtypes[index])
		}
		if length > 1 {
			err := g.setupFor1DFloat32Input(index, input[index])
			if err != nil {
				return err
			}
		} else {
			err := g.setupFor2DFloat32Input(index, input)
			if err != nil {
				return err
			}
		}
	}
	g.RunNodes()
	return nil
}

func (g *Graph) Execute3DFloat32(input [][][]float32) error {
	length := len(g.inputs)
	if length > 1 && length != len(input) {
		return fmt.Errorf("args count not equal, got %d, wanted %d", len(input), length)
	} else if length == 1 {
		return fmt.Errorf("3D input not currently supported")
	}
	for index := range g.inputs {
		if g.dtypes[index] != tensor.Float && g.dtypes[index] != tensor.Double {
			return fmt.Errorf("expected float or double inputs, got %d", g.dtypes[index])
		}
		err := g.setupFor2DFloat32Input(index, input[index])
		if err != nil {
			return err
		}
	}
	g.RunNodes()
	return nil
}
