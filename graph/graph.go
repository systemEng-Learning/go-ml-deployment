package graph

import (
	"fmt"
	"strconv"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
	tensors "github.com/systemEng-Learning/go-ml-deployment/tensor"
)

type Ops interface {
	Init(g *Graph, node *ir.NodeProto) error
	Compute(g *Graph) error
}

type Graph struct {
	graph     *ir.GraphProto
	input     string
	shape     []int
	nodes     []Ops
	tensors   []*tensors.Tensor
	output    []int
	tensorMap map[string]int // map of tensor name to index in tensors slice. Only used temporarily during setup
}

func (g *Graph) Init() {
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
	fmt.Printf("%+v\n", g.nodes)
}

func (g *Graph) setInputsTensor() error {
	g.tensors = make([]*tensors.Tensor, 0)
	g.tensorMap = make(map[string]int)
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
		dtype := tensors.OnnxTypeToDtype(ir.TensorProto_DataType_name[tensor.TensorType.ElemType])
		inputTensor, err := tensors.CreateTensor(shape, dtype)
		if err != nil {
			return err
		}
		g.tensors = append(g.tensors, inputTensor)
		g.tensorMap[input.Name] = i
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
			result[i] = 1
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
			l := &LinearClassifier{}
			err = l.Init(g, node)
			g.nodes = append(g.nodes, l)
		case "Cast":
			c := &Cast{}
			err = c.Init(g, node)
			g.nodes = append(g.nodes, c)
		case "Normalizer":
			n := &Normalizer{}
			err = n.Init(g, node)
			g.nodes = append(g.nodes, n)
		case "ZipMap":
			z := &ZipMap{}
			err = z.Init(g, node)
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

func (g *Graph) GetTensorIndex(name string) (int, error) {
	index, ok := g.tensorMap[name]
	if !ok {
		return -1, fmt.Errorf("tensor with name %s does not exist")
	}
	return index, nil
}

func (g *Graph) RegisterTensor(name string) int {
	index, ok := g.tensorMap[name]
	if !ok {
		g.tensors = append(g.tensors, nil)
		index = len(g.tensors) - 1
		g.tensorMap[name] = index
	}
	return index
}

func (g *Graph) setOutputIndices() error {
	g.output = make([]int, len(g.graph.Output))
	for i, o := range g.graph.Output {
		index, ok := g.tensorMap[o.Name]
		if !ok {
			return fmt.Errorf("output index: this output does not exist")
		}
		g.output[i] = index
	}
	return nil
}

func (g *Graph) Execute(input [][]float32) {
	t := Tensor{Floats: input}
	g.tensors = make(map[string]*Tensor)
	g.tensors[g.input] = &t
	op, ok := g.nodes[g.input]
	if !ok {
		panic("Nope")
	}
	err := op.Compute(g)
	if err != nil {
		panic(err)
	}
}

func (g *Graph) ComputeNext(key string) error {
	op, ok := g.nodes[key]
	if !ok {
		return nil
	}
	err := op.Compute(g)

	if err != nil {
		return err
	}
	return nil
}

func (g *Graph) PrintOutput() {
	for _, t := range g.tensors {
		fmt.Println(g.tensors[o.Name])
	}
}
