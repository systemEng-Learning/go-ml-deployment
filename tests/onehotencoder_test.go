package tests

import (
	"testing"

	"github.com/systemEng-Learning/go-ml-deployment/tensor"
)

func runOHETestIntCategory[T tensor.Numeric](t *testing.T, input []T) {
	categories := []int64{0, 1, 2, 3, 4, 5, 6, 7}
	expected_output := make([][]float32, len(input))
	for i := range input {
		expected_output[i] = make([]float32, len(categories))
		for j := range categories {
			if int64(input[i]) != categories[j] {
				expected_output[i][j] = 0
			} else {
				expected_output[i][j] = 1
			}
		}
	}

	sg := Test("OneHotEncoder")
	sg.addInput("X", []int{7}, input)
	sg.addAttribute("cats_int64", categories)
	sg.addAttribute("zeros", int64(1))
	sg.addOutput("Y", expected_output)
	sg.errorBound = 0.00000001
	sg.Execute(t)

	sg.addAttribute("zeros", int64(0))
	err := sg.Execute(t)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestOHEIntegerWithInt64(t *testing.T) {
	runOHETestIntCategory(t, []int64{8, 1, 0, 0, 3, 7, 4})
}

func TestOHEIntegerWithDouble(t *testing.T) {
	runOHETestIntCategory(t, []float64{8.1, 1.2, 0, 0.7, 3.4, 7.9, 4.4})
}

func TestOHEString(t *testing.T) {
	categories := []string{"Apple", "Orange", "Watermelon", "Blueberry", "Coconut", "Mango", "Tangerine"}
	input := []string{"Watermelon", "Orange", "Tangerine", "Apple", "Kit"}
	expected_output := make([][]float32, len(input))
	for i := range input {
		expected_output[i] = make([]float32, len(categories))
		for j := range categories {
			if input[i] != categories[j] {
				expected_output[i][j] = 0
			} else {
				expected_output[i][j] = 1
			}
		}
	}
	sg := Test("OneHotEncoder")
	sg.addInput("X", []int{5}, input)
	sg.addAttribute("cats_strings", categories)
	sg.addAttribute("zeros", int64(1))
	sg.addOutput("Y", expected_output)
	sg.errorBound = 0.00000001
	sg.Execute(t)

	sg.addAttribute("zeros", int64(0))
	err := sg.Execute(t)
	if err == nil {
		t.Fatal("expected error")
	}
}
