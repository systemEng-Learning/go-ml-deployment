package main

import (
	"fmt"
	"strconv"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
)

type Ops interface {
	Compute(g *Graph) error
}

type Tensor struct {
	F          []float32
	I          []int64
	Floats     [][]float32
	Ints       [][]int64
	IntMaps    []map[int]float32
	StringMaps []map[string]float32
}

type Graph struct {
	graph   *ir.GraphProto
	input   string
	shape   []int
	nodes   map[string]Ops
	tensors map[string]*Tensor
}

func (g *Graph) Init() {
	err := g.SetInputShape()
	if err != nil {
		panic(err)
	}
	g.input = g.graph.Input[0].Name
	err = g.initializeNodes()
	if err != nil {
		panic(err)
	}
	fmt.Printf("%+v\n", g.nodes)
}

func (g *Graph) SetInputShape() error {
	input := g.graph.Input[0]
	v := input.GetType().GetValue()
	tensor := v.(*ir.TypeProto_TensorType)
	shape, err := GetShape(tensor.TensorType.Shape)
	if err != nil {
		return err
	}
	fmt.Println(shape)
	if len(shape) == 1 && shape[0] != -1 {
		shape = append(shape, shape[0])
		shape[0] = 1
	}
	g.shape = shape
	return nil
}

func GetShape(shape *ir.TensorShapeProto) ([]int, error) {
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
	g.nodes = make(map[string]Ops)
	for _, node := range g.graph.Node {
		o := node.OpType
		switch o {
		case "LinearClassifier":
			var l *LinearClassifier
			l, err = NewLinearClassifier(node)
			g.nodes[l.Input] = l
		case "Cast":
			var c *Cast
			c, err = NewCast(node)
			g.nodes[c.Input] = c
		case "Normalizer":
			var n *Normalizer
			n, err = NewNormalizer(node)
			g.nodes[n.Input] = n
		case "ZipMap":
			var z *ZipMap
			z, err = NewZipMap(node)
			g.nodes[z.Input] = z
		default:
			return fmt.Errorf("%s operation not supported", node.OpType)
		}
		if err != nil {
			return err
		}
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
	for _, o := range g.graph.Output {
		fmt.Println(g.tensors[o.Name])
	}
}
