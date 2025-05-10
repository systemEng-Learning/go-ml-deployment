package tests

import (
	"math"
	"os"
	"testing"

	"github.com/systemEng-Learning/go-ml-deployment/graph"
	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"google.golang.org/protobuf/proto"
)

func TestDictVectorizer(t *testing.T) {
	// Load the ONNX model file
	in, err := os.ReadFile("../examples/dv.onnx")
	if err != nil {
		t.Fatalf("Error reading file: %v", err)
	}

	// Parse the ONNX model
	model := &ir.ModelProto{}
	if err := proto.Unmarshal(in, model); err != nil {
		t.Fatalf("Failed to parse model file: %v", err)
	}

	// Initialize the graph
	graphProto := model.GetGraph()
	graph := graph.Graph{}
	if err := graph.Init(graphProto); err != nil {
		t.Fatalf("Error initializing graph: %v", err)
	}

	// Input data
	f := []map[string]float32{
		{"foo": 2.0, "bar": 0.0},
		{"bar": 1.0, "foo": 0.0, "baz": 3.0},
	}

	// Execute the graph
	result, err := graph.Execute([]any{f})
	if err != nil {
		t.Fatalf("Error executing graph: %v", err)
	}

	// Expected output
	expectedOutput := [][]float32{
		{0, 0, 2},
		{1, 3, 0},
	}

	// Validate the output
	if len(result) == 0 {
		t.Fatalf("No output received from graph execution")
	}

	outputTensor, ok := result[0].([][]float32)
	if !ok {
		t.Fatalf("Unexpected output type: %T", result[0])
	}

	if len(outputTensor) != len(expectedOutput) {
		t.Fatalf("Output row count mismatch. Got %d, expected %d", len(outputTensor), len(expectedOutput))
	}

	for i := range outputTensor {
		if len(outputTensor[i]) != len(expectedOutput[i]) {
			t.Fatalf("Output column count mismatch in row %d. Got %d, expected %d", i, len(outputTensor[i]), len(expectedOutput[i]))
		}
		for j := range outputTensor[i] {
			got := outputTensor[i][j]
			want := expectedOutput[i][j]
			if math.Abs(float64(got-want)) > 0.001 {
				t.Errorf("Output mismatch at index %d: got %f, want %f", i, got, want)
			}
		}
	}
}

func TestDictVectorizer1D(t *testing.T) {
	// Load the ONNX model file
	in, err := os.ReadFile("../examples/dv.onnx")
	if err != nil {
		t.Fatalf("Error reading file: %v", err)
	}

	// Parse the ONNX model
	model := &ir.ModelProto{}
	if err := proto.Unmarshal(in, model); err != nil {
		t.Fatalf("Failed to parse model file: %v", err)
	}

	// Initialize the graph
	graphProto := model.GetGraph()
	graph := graph.Graph{}
	if err := graph.Init(graphProto); err != nil {
		t.Fatalf("Error initializing graph: %v", err)
	}

	// Input data
	f := []map[string]float32{
		{"bar": 1.0, "foo": 0.0, "baz": 3.0},
	}

	// Execute the graph
	result, err := graph.Execute([]any{f})
	if err != nil {
		t.Fatalf("Error executing graph: %v", err)
	}

	// Expected output
	expectedOutput := [][]float32{
		{1, 3, 0},
	}

	// Validate the output
	if len(result) == 0 {
		t.Fatalf("No output received from graph execution")
	}

	outputTensor, ok := result[0].([][]float32)
	if !ok {
		t.Fatalf("Unexpected output type: %T", result[0])
	}

	if len(outputTensor) != len(expectedOutput) {
		t.Fatalf("Output row count mismatch. Got %d, expected %d", len(outputTensor), len(expectedOutput))
	}

	for i := range outputTensor {
		if len(outputTensor[i]) != len(expectedOutput[i]) {
			t.Fatalf("Output column count mismatch in row %d. Got %d, expected %d", i, len(outputTensor[i]), len(expectedOutput[i]))
		}
		for j := range outputTensor[i] {
			got := outputTensor[i][j]
			want := expectedOutput[i][j]
			if math.Abs(float64(got-want)) > 0.001 {
				t.Errorf("Output mismatch at index %d: got %f, want %f", i, got, want)
			}
		}
	}
}
