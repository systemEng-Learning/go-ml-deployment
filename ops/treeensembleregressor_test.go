package ops_test

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/systemEng-Learning/go-ml-deployment/graph"
	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"google.golang.org/protobuf/proto"
)

func TestTreeEnsembleRegressor(t *testing.T) {
	// Load the ONNX model file
	in, err := os.ReadFile("../examples/dtr_diabetes.onnx") // Replace with the actual ONNX model file path
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
		{0.04534098, -0.04464164, -0.00620595, -0.01599898, 0.1250187,
			0.1251981, 0.019187, 0.03430886, 0.03243232, -0.0052198},
		{0.09256398, -0.04464164, 0.03690653, 0.02187239, -0.02496016,
			-0.01665815, 0.00077881, -0.03949338, -0.02251653, -0.02178823},
		{0.06350368, 0.05068012, -0.00405033, -0.01255612, 0.10300346,
			0.04878988, 0.05600338, -0.00259226, 0.08449153, -0.01764613},
		{0.09619652, -0.04464164, 0.0519959, 0.07926471, 0.05484511,
			0.03657709, -0.07653559, 0.14132211, 0.09864806, 0.06105391},
	}

	result, err := graph.Execute([]any{inputData})
	if err != nil {
		t.Fatalf("Error executing graph: %v", err)
	}

	if len(result) == 0 {
		t.Fatalf("No output received from graph execution")
	}

	expectedOutput := [][]float32{
		{190},
		{288},
		{170},
		{277},
	}

	output, ok := result[0].([][]float32)
	if !ok {
		t.Fatalf("Unexpected output type: %T", result[0])
	}
	fmt.Println("outputbvv:", output)

	if len(output) != len(expectedOutput) {
		t.Fatalf("Output length mismatch. Got %d, expected %d", len(output), len(expectedOutput))
	}

	for i := range output {
		for j := range output[i] {
			got := output[i][j]
			want := expectedOutput[i][j]
			if math.Abs(float64(got-want)) > 0.001 {
				t.Errorf("Output mismatch at index %d: got %f, want %f", i, got, want)
			}
		}
	}
}
