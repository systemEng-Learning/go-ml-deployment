package graph

import (
	"fmt"
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
		t := v.(*ir.TypeProto_TensorType)
		shape, err := getShape(t.TensorType.Shape)
		if err != nil {
			return err
		}
		if len(shape) == 1 && shape[0] != -1 {
			shape = append(shape, shape[0])
			shape[0] = 1
		} else if len(shape) > 2 {
			return fmt.Errorf("graph setinputtensor: want inputs of at most 2 dimensions, got %d", len(shape))
		}
		dtype := tensors.OnnxTypeToDtype(t.TensorType.ElemType)
		index := g.kernel.RegisterWriter(input.Name)
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
		case "LinearRegressor":
			l := &ops.LinearRegressor{}
			err = l.Init(g.kernel, node)
			g.nodes = append(g.nodes, l)
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
		index, err := g.kernel.RegisterReader(o.Name)
		if err != nil {
			return err
		}
		g.output[i] = index
	}
	return nil
}

func (g *Graph) RunNodes() {
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
