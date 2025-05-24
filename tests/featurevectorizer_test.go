package tests

import (
	"fmt"
	"testing"
)

func TestFeatureVectorizerBasic(t *testing.T) {
	fv := Test("FeatureVectorizer")
	fmt.Println(fv)
	fv.addAttribute("input_dimensions", []int64{3, 2, 1, 4})

	fv.addInput("X", []int{1, 3}, []int64{1, 2, 3})
	fv.addInput("X1", []int{1, 2}, []int64{4, 5})
	fv.addInput("X2", []int{1}, []int64{6})
	fv.addInput("X3", []int{1, 4}, []int64{7, 8, 9, 10})
	fv.addOutput("Y", [][]int64{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}})
	err := fv.Execute(t)
	if err != nil {
		t.Fatalf("error shouldn't exist: %v", err)
	}

}

// Test inputDimensions mismatch
func TestFeatureVectorizerMisMatch(t *testing.T) {
	fv := Test("FeatureVectorizer")
	fmt.Println(fv)
	fv.addAttribute("input_dimensions", []int64{2, 3})

	fv.addInput("X", []int{1, 3}, []int64{1, 2, 3})
	fv.addInput("X1", []int{1, 2}, []int64{1, 2})
	fv.addOutput("Y", [][]int64{{1, 2, 1, 2, 0}})
	err := fv.Execute(t)
	if err != nil {
		t.Fatalf("error shouldn't exist: %v", err)
	}

}

func TestFeatureVectorizerBatch(t *testing.T) {
	fv := Test("FeatureVectorizer")
	fmt.Println(fv)
	fv.addAttribute("input_dimensions", []int64{2, 2})

	fv.addInput("X", []int{2, 2}, []float64{1.0, 2.0, 3.0, 4.0})
	fv.addInput("X1", []int{2, 2}, []float64{10.0, 11.0, 12.0, 13.0})
	fv.addOutput("Y", [][]float64{{1.0, 2.0, 10.0, 11.0},
		{3.0, 4.0, 12.0, 13.0}})
	fv.errorBound = 0.00001
	err := fv.Execute(t)
	if err != nil {
		t.Fatalf("error shouldn't exist: %v", err)
	}

}
