package ops_test

import (
	"os"
	"testing"

	"github.com/systemEng-Learning/go-ml-deployment/graph"
	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"google.golang.org/protobuf/proto"
)

func TestTreeEnsembleClassifier(t *testing.T) {
	// Load the ONNX model file
	in, err := os.ReadFile("../examples/dtc_iris.onnx")
	if err != nil {
		t.Fatalf("Error reading file: %v", err)
	}

	// Parse the model
	model := &ir.ModelProto{}
	if err := proto.Unmarshal(in, model); err != nil {
		t.Fatalf("Failed to parse model file: %v", err)
	}

	// Initialize the graph
	graphProto := model.GetGraph()
	graph := graph.Graph{}
	graph.Init(graphProto)

	// Input data
	inputData := [][]float32{
		{6.0, 3.4, 4.5, 1.6},
		{5.7, 3.8, 1.7, 0.3},
		{7.7, 2.6, 6.9, 2.3},
		{6.0, 2.9, 4.5, 1.5},
	}

	// Execute the graph
	result, err := graph.Execute([]any{inputData})
	if err != nil {
		t.Fatalf("Error executing graph: %v", err)
	}

	// Check the output
	if len(result) == 0 {
		t.Fatalf("No output received from graph execution")
	}

	// Expected output
	expectedOutput := []int64{1, 0, 2, 1}

	// Validate the output
	output, ok := result[0].([]int64)
	if !ok {
		t.Fatalf("Unexpected output type: %T", result[0])
	}

	if len(output) != len(expectedOutput) {
		t.Fatalf("Output length mismatch. Got %d, expected %d", len(output), len(expectedOutput))
	}

	for i, val := range output {
		if val != expectedOutput[i] {
			t.Errorf("Output mismatch at index %d. Got %d, expected %d", i, val, expectedOutput[i])
		}
	}
}
