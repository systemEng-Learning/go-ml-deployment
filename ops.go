package main

import (
	"fmt"
	"math"

	"github.com/systemEng-Learning/go-ml-deployment/ir"
)

type LinearClassifier struct {
	Input             string
	classlabel        []int64
	classlabel_string [][]byte
	coefficients      []float32
	intercepts        []float32
	multiclass        bool
	post_transform    string
	outputs           []string
}

func NewLinearClassifier(node *ir.NodeProto) (*LinearClassifier, error) {
	l := &LinearClassifier{
		Input:          node.Input[0],
		multiclass:     false,
		post_transform: "NONE",
		outputs:        node.Output,
	}
	for _, attr := range node.Attribute {
		switch attr.Name {
		case "classlabels_ints":
			l.classlabel = attr.Ints
		case "classlabels_strings":
			l.classlabel_string = attr.Strings
		case "coefficients":
			l.coefficients = attr.Floats
		case "intercepts":
			l.intercepts = attr.Floats
		case "multi_class":
			if attr.I > 0 {
				l.multiclass = true
			}
		case "post_transform":
			l.post_transform = string(attr.S)
		default:
			return nil, fmt.Errorf("%s not supported for %s", attr.Name, node.OpType)
		}
	}
	return l, nil
}

func (l *LinearClassifier) Compute(g *Graph) error {
	if len(g.shape) != 2 {
		return fmt.Errorf("linearclassifier: invalid shape %v", g.shape)
	}
	t, ok := g.tensors[l.Input]
	if !ok {
		return fmt.Errorf("linearclassifier: no input")
	}
	x := t.Floats
	s := len(x[0])
	if s != g.shape[1] {
		return fmt.Errorf("each input shape %d isn't the same as expected column %d", s, g.shape[1])
	}
	num_class := len(l.coefficients) / s
	scores := make([][]float32, len(x))
	for d := range x {
		scores[d] = make([]float32, num_class)
		for i := 0; i < num_class; i++ {
			for j := 0; j < s; j++ {
				scores[d][i] += x[d][j] * l.coefficients[i*s+j]
			}
		}
	}

	if l.intercepts != nil {
		for i := range scores {
			for j := range scores[i] {
				scores[i][j] += l.intercepts[j]
			}
		}
	}

	labels := make([]int64, len(x))
	for d := range scores {
		max_class := 0
		max_weight := scores[d][0]

		for i := 1; i < num_class; i++ {
			if scores[d][i] > max_weight {
				max_class = i
				max_weight = scores[d][i]
			}
		}
		labels[d] = l.classlabel[max_class]
	}

	if l.post_transform == "SOFTMAX" {
		computeSoftmaxInplace(scores)
	}

	g.tensors[l.outputs[0]] = &Tensor{I: labels}
	g.tensors[l.outputs[1]] = &Tensor{Floats: scores}

	var err error
	for _, o := range l.outputs {
		err = g.ComputeNext(o)
		if err != nil {
			return err
		}
	}

	return nil
}

func computeSoftmaxInplace(scores [][]float32) {
	var sum float32
	for i := range scores {
		sum = 0
		for j := range scores[i] {
			scores[i][j] = float32(math.Exp(float64(scores[i][j])))
			sum += scores[i][j]
		}

		for j := range scores[i] {
			scores[i][j] = scores[i][j] / sum
		}
	}
}

type Cast struct {
	Input    string
	output   string
	to       int64
	saturate bool
}

func NewCast(node *ir.NodeProto) (*Cast, error) {
	c := &Cast{
		Input:    node.Input[0],
		saturate: true,
		output:   node.Output[0],
	}
	for _, attr := range node.Attribute {
		switch attr.Name {
		case "to":
			c.to = attr.I
		case "saturate":
			if attr.I == 0 {
				c.saturate = false
			}
		default:
			return nil, fmt.Errorf("%s not supported for %s", attr.Name, node.OpType)
		}
	}
	return c, nil
}

func (c *Cast) Compute(g *Graph) error {
	t := g.tensors[c.Input]
	if t.I != nil {
		g.tensors[c.output] = t
	}
	err := g.ComputeNext(c.output)
	return err
}

type Normalizer struct {
	Input  string
	output string
	norm   string
}

func NewNormalizer(node *ir.NodeProto) (*Normalizer, error) {
	n := &Normalizer{
		Input:  node.Input[0],
		output: node.Output[0],
	}
	for _, attr := range node.Attribute {
		switch attr.Name {
		case "norm":
			n.norm = string(attr.S)
		default:
			return nil, fmt.Errorf("%s not supported for %s", attr.Name, node.OpType)
		}
	}
	return n, nil
}

func (n *Normalizer) Compute(g *Graph) error {
	t := g.tensors[n.Input]
	scores := t.Floats
	var sum float32
	for i := range scores {
		sum = 0
		for j := range scores[i] {
			sum += scores[i][j]
		}

		for j := range scores[i] {
			scores[i][j] = scores[i][j] / sum
		}
	}
	g.tensors[n.output] = &Tensor{Floats: scores}

	return g.ComputeNext(n.output)
}

type ZipMap struct {
	Input               string
	classlabels_int64s  []int64
	classlabels_strings [][]byte
	output              string
}

func NewZipMap(node *ir.NodeProto) (*ZipMap, error) {
	z := &ZipMap{
		Input:  node.Input[0],
		output: node.Output[0],
	}
	for _, attr := range node.Attribute {
		switch attr.Name {
		case "classlabels_int64s":
			z.classlabels_int64s = attr.Ints
		case "classlabels_strings":
			z.classlabels_strings = attr.Strings
		default:
			return nil, fmt.Errorf("%s not supported for %s", attr.Name, node.OpType)
		}
	}
	return z, nil
}

func (z *ZipMap) Compute(g *Graph) error {
	t := g.tensors[z.Input]
	scores := t.Floats
	result := make([]map[int]float32, len(scores))
	for i := range scores {
		m := make(map[int]float32)
		for j := range scores[i] {
			m[int(z.classlabels_int64s[j])] = scores[i][j]
		}
		result[i] = m
	}
	g.tensors[z.output] = &Tensor{IntMaps: result}

	return g.ComputeNext(z.output)
}
