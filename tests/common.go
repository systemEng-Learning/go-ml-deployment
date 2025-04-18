package tests

import (
	"log"
	"math"
	"reflect"
	"testing"

	"github.com/systemEng-Learning/go-ml-deployment/graph"
	"github.com/systemEng-Learning/go-ml-deployment/ir"
)

type SingleNodeGraph struct {
	graph      *ir.GraphProto
	inputs     []any
	outputs    []any
	errorBound float64
}

func Test(nodeName string) *SingleNodeGraph {
	sg := SingleNodeGraph{}
	sg.graph = &ir.GraphProto{}
	sg.graph.Node = append(sg.graph.Node, &ir.NodeProto{OpType: nodeName})
	return &sg
}

func (sg *SingleNodeGraph) addAttribute(name string, value any) {
	attr := ir.AttributeProto{Name: name}
	switch item := value.(type) {
	case int64:
		attr.I = item
	case []byte:
		attr.S = item
	case []int64:
		attr.Ints = item
	case []float32:
		attr.Floats = item
	case [][]byte:
		attr.Strings = item
	default:
		log.Fatalf("unsupported type for %v", item)
	}
	sg.graph.Node[0].Attribute = append(sg.graph.Node[0].Attribute, &attr)
}

func (sg *SingleNodeGraph) addInput(name string, shape []int, value any) {
	sg.inputs = append(sg.inputs, value)
	input := ir.ValueInfoProto{Name: name}
	tt := ir.TypeProto_TensorType{}
	var elemType int32
	switch value.(type) {
	case int32, []int32, [][]int32:
		elemType = ir.TensorProto_DataType_value["INT32"]
	case int64, []int64, [][]int64:
		elemType = ir.TensorProto_DataType_value["INT64"]
	case float32, []float32, [][]float32:
		elemType = ir.TensorProto_DataType_value["FLOAT"]
	case float64, []float64, [][]float64:
		elemType = ir.TensorProto_DataType_value["DOUBLE"]
	case string, []string, [][]string:
		elemType = ir.TensorProto_DataType_value["STRING"]
	default:
		elemType = 0
	}
	tt.TensorType = &ir.TypeProto_Tensor{Shape: &ir.TensorShapeProto{}, ElemType: elemType}
	tt.TensorType.Shape.Dim = make([]*ir.TensorShapeProto_Dimension, 0)
	for i := range shape {
		d := ir.TensorShapeProto_Dimension{Value: &ir.TensorShapeProto_Dimension_DimValue{DimValue: int64(shape[i])}}
		tt.TensorType.Shape.Dim = append(tt.TensorType.Shape.Dim, &d)
	}
	input.Type = &ir.TypeProto{}
	input.Type.Value = &tt
	sg.graph.Input = append(sg.graph.Input, &input)
	sg.graph.Node[0].Input = append(sg.graph.Node[0].Input, name)
}

func (sg *SingleNodeGraph) addOutput(name string, value any) {
	output := ir.ValueInfoProto{Name: name}
	sg.graph.Output = append(sg.graph.Output, &output)
	sg.graph.Node[0].Output = append(sg.graph.Node[0].Output, name)
	sg.outputs = append(sg.outputs, value)
}

func (sg *SingleNodeGraph) Execute(t *testing.T) {
	graph := graph.Graph{}
	graph.Init(sg.graph)
	outputs, err := graph.Execute(sg.inputs)
	if err != nil {
		t.Fatalf("error thrown while executing graph: %v", err)
	}
	for i := range outputs {
		switch item := outputs[i].(type) {
		case []int64:
			o := sg.outputs[i].([]int64)
			if !reflect.DeepEqual(o, item) {
				t.Fatalf("expected %v, got %v", o, item)
			}
		case []string:
			o := sg.outputs[i].([]string)
			if !reflect.DeepEqual(o, item) {
				t.Fatalf("expected %v, got %v", o, item)
			}
		case []float32:
			o := sg.outputs[i].([]float32)
			for x := range o {
				if math.Abs(float64(o[x]-item[x])) >= sg.errorBound {
					t.Fatalf("expected %v, got %v", o, item)
				}
			}
		case []float64:
			o := sg.outputs[i].([]float64)
			for x := range o {
				if math.Abs(o[x]-item[x]) >= sg.errorBound {
					t.Fatalf("expected %v, got %v", o, item)
				}
			}
		case [][]int64:
			o := sg.outputs[i].([][]int64)
			if !reflect.DeepEqual(o, item) {
				t.Fatalf("expected %v, got %v", o, item)
			}
		case [][]float32:
			o := sg.outputs[i].([][]float32)
			for x := range o {
				for y := range o[x] {
					if math.Abs(float64(o[x][y]-item[x][y])) >= sg.errorBound {
						t.Fatalf("expected %v, got %v", o, item)
					}
				}
			}
		case [][]float64:
			o := sg.outputs[i].([][]float64)
			for x := range o {
				for y := range o[x] {
					if math.Abs(o[x][y]-item[x][y]) >= sg.errorBound {
						t.Fatalf("expected %v, got %v", o, item)
					}
				}
			}
		}
	}
}
