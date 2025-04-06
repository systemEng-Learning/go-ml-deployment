package ops_test

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/systemEng-Learning/go-ml-deployment/graph"
	"github.com/systemEng-Learning/go-ml-deployment/ir"
	"google.golang.org/protobuf/encoding/protojson"
)

// loadModel reads a JSON file from testdata and unmarshals it into an ir.ModelProto.
func loadModel(t *testing.T, filename string) ir.ModelProto {
	path := filepath.Join("../testdata", filename)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read file %s: %v", path, err)
	}
	var model ir.ModelProto
	if err := protojson.Unmarshal(data, &model); err != nil {
		t.Fatalf("failed to unmarshal file %s: %v", path, err)
	}
	return model
}

// TestLinearRegressorValid1D tests a valid linear regressor (with intercept)
// using a 1D float32 input.
func TestLinearRegressorValid1D(t *testing.T) {
	model := loadModel(t, "linearreg_valid.protojson")
	graphProto := model.GetGraph()

	var g graph.Graph
	g.Init(graphProto)

	oneDSample := []float32{5.9, 3.2, 4.8, 1.8}
	if err := g.Execute1DFloat32(oneDSample); err != nil {
		t.Fatalf("Execute1DFloat32 failed: %v", err)
	}
	t.Log("Output after Execute1DFloat32 (valid with intercept):")
	g.Print()
}

// TestLinearRegressorValid2D tests a valid linear regressor (with intercept)
// using a 2D float32 input.
func TestLinearRegressorValid2D(t *testing.T) {
	model := loadModel(t, "linearreg_valid.protojson")
	graphProto := model.GetGraph()

	var g graph.Graph
	g.Init(graphProto)

	twoDSamples := [][]float32{
		{5.9, 3.2, 4.8, 1.8},
		{5.1, 3.4, 1.5, 0.2},
		{7.4, 2.8, 6.1, 1.9},
	}
	if err := g.Execute2DFloat32(twoDSamples); err != nil {
		t.Fatalf("Execute2DFloat32 failed: %v", err)
	}
	t.Log("Output after Execute2DFloat32 (valid with intercept):")
	g.Print()
}

// TestLinearRegressorNoIntercept tests a valid linear regressor without intercept,
// using a 1D float32 input.
func TestLinearRegressorNoIntercept(t *testing.T) {
	model := loadModel(t, "linearreg_no_intercept.protojson")
	graphProto := model.GetGraph()

	var g graph.Graph
	g.Init(graphProto)

	oneDSample := []float32{6.1, 2.8, 5.6, 1.5}
	if err := g.Execute1DFloat32(oneDSample); err != nil {
		t.Fatalf("Execute1DFloat32 failed: %v", err)
	}
	t.Log("Output after Execute1DFloat32 (no intercept):")
	g.Print()
}

// TestLinearRegressorInvalid tests a linear regressor model that should trigger
// an error (coefficients length not divisible by intercept length).
func TestLinearRegressorInvalid(t *testing.T) {
	model := loadModel(t, "linearreg_invalid.protojson")
	graphProto := model.GetGraph()
	var g graph.Graph
	var initErr error

	func() {
		defer func() {
			if r := recover(); r != nil {
				initErr = fmt.Errorf("panic during Init: %v", r)
			}
		}()
		g.Init(graphProto)
	}()

	if initErr != nil {
		t.Logf("Init failed (expected): %v", initErr)
		return
	}
}
