package tests

import "testing"

func runGatherTest2DOutput[T NumericType](t *testing.T, input [][]T, inputShape []int, indices []int64, indicesShape []int64, expected [][]T, axes ...int64) {
	sg := Test("Gather")
	sg.addInput("X", inputShape, input)
	sg.addInitializer("indices", indicesShape, indices)
	sg.addOutput("Y", expected)
	if len(axes) > 0 {
		sg.addAttribute("axis", axes[0])
	}
	sg.errorBound = 0.0000001
	sg.Execute(t)
}

func TestGatherInt(t *testing.T) {
	runGatherTest2DOutput(t, [][]int64{{0, 1}, {2, 3}}, []int{2, 2}, []int64{0}, []int64{1}, [][]int64{{0, 1}})
	runGatherTest2DOutput(
		t,
		[][]float32{{0.0, 1.0, 2.0, 3.0}, {4.0, 5.0, 6.0, 7.0}, {8.0, 9.0, 10.0, 11.0}},
		[]int{3, 4}, []int64{1, 0}, []int64{2},
		[][]float32{{4.0, 5.0, 6.0, 7.0}, {0.0, 1.0, 2.0, 3.0}},
	)
	runGatherTest2DOutput(
		t,
		[][]float32{{0.0, 1.0, 2.0, 3.0}, {4.0, 5.0, 6.0, 7.0}, {8.0, 9.0, 10.0, 11.0}},
		[]int{3, 4}, []int64{0, 3}, []int64{2},
		[][]float32{{0.0, 3.0}, {4.0, 7.0}, {8.0, 11.0}},
		-1,
	)
}
