package graph

import (
	"fmt"
	"slices"
	"strconv"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"github.com/systemEng-Learning/go-ml-deployment/kernel"
	"github.com/systemEng-Learning/go-ml-deployment/ops"
	tensors "github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type Ops interface {
	Init(g *kernel.Kernel, node *ir.NodeProto) error
	Compute(g *kernel.Kernel) error
}

type Graph struct {
	graph  *ir.GraphProto
	inputs []int
	shapes [][]int
	dtypes []tensors.DataType
	nodes  []Ops
	output []int
	kernel *kernel.Kernel
}

func (g *Graph) Init(graphProto *ir.GraphProto) {
	g.graph = graphProto
	g.kernel = &kernel.Kernel{}
	g.kernel.Init()
	err := g.setInputsTensor()
	if err != nil {
		panic(err)
	}
	err = g.initializeNodes()
	if err != nil {
		panic(err)
	}
	err = g.setOutputIndices()
	if err != nil {
		panic(err)
	}
}

func (g *Graph) setInputsTensor() error {
	g.inputs = make([]int, len(g.graph.Input))
	g.shapes = make([][]int, len(g.graph.Input))
	g.dtypes = make([]tensors.DataType, len(g.graph.Input))
	for i, input := range g.graph.Input {
		v := input.GetType().GetValue()
		tensor := v.(*ir.TypeProto_TensorType)
		shape, err := getShape(tensor.TensorType.Shape)
		if err != nil {
			return err
		}
		if len(shape) == 1 && shape[0] != -1 {
			shape = append(shape, shape[0])
			shape[0] = 1
		} else if len(shape) > 2 {
			return fmt.Errorf("graph setinputtensor: want inputs of at most 2 dimensions, got %d", len(shape))
		}
		dtype := tensors.OnnxTypeToDtype(tensor.TensorType.ElemType)
		index := g.kernel.RegisterTensor(input.Name)
		g.inputs[i] = index
		g.shapes[i] = shape
		g.dtypes[i] = dtype
	}
	return nil
}

func getShape(shape *ir.TensorShapeProto) ([]int, error) {
	if shape == nil {
		fmt.Println("No shape")
		return []int{-1}, nil
	}
	result := make([]int, len(shape.Dim))
	for i, d := range shape.Dim {
		v := d.Value
		switch t := v.(type) {
		case *ir.TensorShapeProto_Dimension_DimParam:
			e, err := strconv.ParseInt(t.DimParam, 10, 32)
			if err != nil {
				return result, err
			}
			result[i] = int(e)
		case *ir.TensorShapeProto_Dimension_DimValue:
			result[i] = int(t.DimValue)
		default:
			result[i] = -1
		}
	}
	return result, nil
}

func (g *Graph) initializeNodes() error {
	var err error
	g.nodes = make([]Ops, 0)
	for _, node := range g.graph.Node {
		o := node.OpType
		switch o {
		case "LinearClassifier":
			l := &ops.LinearClassifier{}
			err = l.Init(g.kernel, node)
			g.nodes = append(g.nodes, l)
		case "Cast":
			c := &ops.Cast{}
			err = c.Init(g.kernel, node)
			g.nodes = append(g.nodes, c)
		case "Normalizer":
			n := &ops.Normalizer{}
			err = n.Init(g.kernel, node)
			g.nodes = append(g.nodes, n)
		case "ZipMap":
			z := &ops.ZipMap{}
			err = z.Init(g.kernel, node)
			g.nodes = append(g.nodes, z)
		default:
			return fmt.Errorf("%s operation not supported", node.OpType)
		}
		if err != nil {
			return err
		}
	}
	return nil
}

func (g *Graph) setOutputIndices() error {
	g.output = make([]int, len(g.graph.Output))
	for i, o := range g.graph.Output {
		index, err := g.kernel.GetTensorIndex(o.Name)
		if err != nil {
			return err
		}
		g.output[i] = index
	}
	return nil
}

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
	tensor, err := g.kernel.Output(g.inputs[index], shape, g.dtypes[index])
	if err != nil {
		return err
	}
	if tensor.DType == tensors.Double {
		for i := range data {
			tensor.DoubleData[i] = float64(data[i])
		}
	} else {
		copy(tensor.FloatData, data)
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
	tensor, err := g.kernel.Output(g.inputs[index], shape, g.dtypes[index])
	if err != nil {
		return err
	}
	if tensor.DType == tensors.Double {
		for x := range m {
			for y := range n {
				tensor.DoubleData[x*n+y] = float64(data[x][y])
			}
		}
	} else {
		for x := range m {
			for y := range n {
				tensor.FloatData[x*n+y] = data[x][y]
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
		if g.dtypes[i] != tensors.Float && g.dtypes[i] != tensors.Double {
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
			tensor, err := g.kernel.Output(g.inputs[i], shape, g.dtypes[i])
			if err != nil {
				return err
			}
			if tensor.DType == tensors.Double {
				tensor.DoubleData[0] = float64(input[i])
			} else {
				tensor.FloatData[0] = input[i]
			}
		} else {
			err := g.setupFor1DFloat32Input(i, input)
			if err != nil {
				return err
			}
		}
	}
	g.execute()
	return nil
}

func (g *Graph) Execute2DFloat32(input [][]float32) error {
	// setups the input tensor
	length := len(g.inputs)
	if length > 1 && length != len(input) {
		return fmt.Errorf("args count not equal, got %d, wanted %d", len(input), length)
	}
	for index := range g.inputs {
		if g.dtypes[index] != tensors.Float && g.dtypes[index] != tensors.Double {
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
	g.execute()
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
		if g.dtypes[index] != tensors.Float && g.dtypes[index] != tensors.Double {
			return fmt.Errorf("expected float or double inputs, got %d", g.dtypes[index])
		}
		err := g.setupFor2DFloat32Input(index, input[index])
		if err != nil {
			return err
		}
	}
	g.execute()
	return nil
}

func (g *Graph) execute() {
	for _, node := range g.nodes {
		err := node.Compute(g.kernel)
		if err != nil {
			panic(err)
		}
	}
}

func (g *Graph) Print() {
	for _, o := range g.output {
		fmt.Println(g.kernel.Get(o))
	}
}
