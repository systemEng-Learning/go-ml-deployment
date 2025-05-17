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
	onnxGraph     *ir.GraphProto
	inputs        []any
	expected      []any
	errorBound    float64
	isRelativeErr bool
	graph         *graph.Graph
}

func Test(nodeName string) *SingleNodeGraph {
	sg := SingleNodeGraph{}
	sg.onnxGraph = &ir.GraphProto{}
	sg.onnxGraph.Node = append(sg.onnxGraph.Node, &ir.NodeProto{OpType: nodeName})
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
	case []string:
		attr.Strings = make([][]byte, len(item))
		for i := range item {
			attr.Strings[i] = []byte(item[i])
		}
	case [][]byte:
		attr.Strings = item
	default:
		log.Fatalf("unsupported type for %v", item)
	}
	sg.onnxGraph.Node[0].Attribute = append(sg.onnxGraph.Node[0].Attribute, &attr)
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
	sg.onnxGraph.Input = append(sg.onnxGraph.Input, &input)
	sg.onnxGraph.Node[0].Input = append(sg.onnxGraph.Node[0].Input, name)
}

func (sg *SingleNodeGraph) addInputMap(name string, value any) {
	sg.inputs = append(sg.inputs, value)
	input := ir.ValueInfoProto{Name: name}
	tt := ir.TypeProto_MapType{}
	var keyType, elemType int32
	switch value.(type) {
	case []map[int64]float32:
		keyType = ir.TensorProto_DataType_value["INT64"]
		elemType = ir.TensorProto_DataType_value["FLOAT"]
	case []map[int64]float64:
		keyType = ir.TensorProto_DataType_value["INT64"]
		elemType = ir.TensorProto_DataType_value["DOUBLE"]
	case []map[int64][]byte, []map[int64]string:
		keyType = ir.TensorProto_DataType_value["INT64"]
		elemType = ir.TensorProto_DataType_value["STRING"]
	case []map[string]float32:
		keyType = ir.TensorProto_DataType_value["STRING"]
		elemType = ir.TensorProto_DataType_value["FLOAT"]
	case []map[string]float64:
		keyType = ir.TensorProto_DataType_value["STRING"]
		elemType = ir.TensorProto_DataType_value["DOUBLE"]
	case []map[string]int64:
		keyType = ir.TensorProto_DataType_value["STRING"]
		elemType = ir.TensorProto_DataType_value["INT64"]
	}

	tt.MapType = &ir.TypeProto_Map{KeyType: keyType, ValueType: &ir.TypeProto{}}
	ss := ir.TypeProto_TensorType{}
	ss.TensorType = &ir.TypeProto_Tensor{Shape: &ir.TensorShapeProto{}, ElemType: elemType}
	ss.TensorType.Shape.Dim = make([]*ir.TensorShapeProto_Dimension, 0)
	d := ir.TensorShapeProto_Dimension{Value: &ir.TensorShapeProto_Dimension_DimValue{DimValue: int64(1)}}
	ss.TensorType.Shape.Dim = append(ss.TensorType.Shape.Dim, &d)
	tt.MapType.ValueType.Value = &ss
	input.Type = &ir.TypeProto{}
	input.Type.Value = &tt
	sg.onnxGraph.Input = append(sg.onnxGraph.Input, &input)
	sg.onnxGraph.Node[0].Input = append(sg.onnxGraph.Node[0].Input, name)
}

func (sg *SingleNodeGraph) setInput(index int, value any) {
	sg.inputs[index] = value
}

func (sg *SingleNodeGraph) addOutput(name string, value any) {
	output := ir.ValueInfoProto{Name: name}
	sg.onnxGraph.Output = append(sg.onnxGraph.Output, &output)
	sg.onnxGraph.Node[0].Output = append(sg.onnxGraph.Node[0].Output, name)
	sg.expected = append(sg.expected, value)
}

func (sg *SingleNodeGraph) InitOnly() error {
	graph := graph.Graph{}
	err := graph.Init(sg.onnxGraph)
	if err != nil {
		return err
	}
	sg.graph = &graph
	return nil
}

func (sg *SingleNodeGraph) RunOnly(t testing.TB, bench bool) error {
	outputs, err := sg.graph.Execute(sg.inputs)
	if err != nil {
		return err
	}
	if bench {
		return nil
	}
	for i := range outputs {
		switch item := outputs[i].(type) {
		case []int64:
			o := sg.expected[i].([]int64)
			if !reflect.DeepEqual(o, item) {
				t.Fatalf("expected %v, got %v", o, item)
			}
		case []string:
			o := sg.expected[i].([]string)
			if !reflect.DeepEqual(o, item) {
				t.Fatalf("expected %v, got %v", o, item)
			}
		case []float32:
			o := sg.expected[i].([]float32)
			for x := range o {
				var err float64 = sg.errorBound
				if sg.isRelativeErr {
					err *= math.Abs(float64(o[x]))
				}
				if math.Abs(float64(o[x]-item[x])) >= err {
					t.Fatalf("expected %v, got %v", o, item)
				}
			}
		case []float64:
			o := sg.expected[i].([]float64)
			for x := range o {
				var err float64 = sg.errorBound
				if sg.isRelativeErr {
					err *= math.Abs(o[x])
				}
				if math.Abs(o[x]-item[x]) >= err {
					t.Fatalf("expected %v, got %v", o, item)
				}
			}
		case [][]int64:
			o := sg.expected[i].([][]int64)
			if !reflect.DeepEqual(o, item) {
				t.Fatalf("expected %v, got %v", o, item)
			}
		case [][]float32:
			o := sg.expected[i].([][]float32)
			for x := range o {
				for y := range o[x] {
					var err float64 = sg.errorBound
					if sg.isRelativeErr {
						err *= math.Abs(float64(o[x][y]))
					}
					if math.Abs(float64(o[x][y]-item[x][y])) >= err {
						t.Fatalf("expected %v, got %v", o, item)
					}
				}
			}
		case [][]float64:
			o := sg.expected[i].([][]float64)
			for x := range o {
				for y := range o[x] {
					var err float64 = sg.errorBound
					if sg.isRelativeErr {
						err *= math.Abs(o[x][y])
					}
					if math.Abs(o[x][y]-item[x][y]) >= err {
						t.Fatalf("expected %v, got %v", o, item)
					}
				}
			}
		case []map[int]float32:
			o := sg.expected[i].([]map[int]float32)
			for x := range o {
				for k, v := range o[x] {
					var err float64 = sg.errorBound
					if sg.isRelativeErr {
						err *= math.Abs(float64(v))
					}
					if math.Abs(float64(v-item[x][k])) >= err {
						t.Fatalf("expected %v, got %v", o, item)
					}
				}
			}
		case []map[string]float32:
			o := sg.expected[i].([]map[string]float32)
			for x := range o {
				for k, v := range o[x] {
					var err float64 = sg.errorBound
					if sg.isRelativeErr {
						err *= math.Abs(float64(v))
					}
					if math.Abs(float64(v-item[x][k])) >= err {
						t.Fatalf("expected %v, got %v", o, item)
					}
				}
			}
		}
	}
	return nil
}

func (sg *SingleNodeGraph) Execute(t testing.TB) error {
	err := sg.InitOnly()
	if err != nil {
		return err
	}
	err = sg.RunOnly(t, false)
	return err
}
