package tests

import "testing"

func TestCast(t *testing.T) {
	sg := Test("Cast")
	to := int64(7)
	inputs := [][]float32{{1.2, 0.7}, {5.76, 197.82}, {15.36, 11.3}}
	predictions := [][]int64{{1, 0}, {5, 197}, {15, 11}}
	sg.addAttribute("to", to)
	sg.addInput("X", []int{3, 2}, inputs)
	sg.addOutput("Y", predictions)
	sg.Execute(t)
}

func TestSameCast(t *testing.T) {
	sg := Test("Cast")
	to := int64(1)
	inputs := [][]float32{{1.2, 0.7}, {5.76, 197.82}, {15.36, 11.3}}
	predictions := [][]float32{{1.2, 0.7}, {5.76, 197.82}, {15.36, 11.3}}
	sg.addAttribute("to", to)
	sg.addInput("X", []int{3, 2}, inputs)
	sg.addOutput("Y", predictions)
	sg.errorBound = 0.00001
	sg.Execute(t)
}
