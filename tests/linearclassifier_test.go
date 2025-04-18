package tests

import "testing"

type ValidInputTypes interface {
	int32 | int64 | float32 | float64
}

func TestLinearClassifierMultiClass(t *testing.T) {
	sg := Test("LinearClassifier")
	coefficients := []float32{-0.22562418, 0.34188559, 0.68346153,
		-0.68051993, -0.1975279, 0.03748541}
	classes := []int64{1, 2, 3}
	multiclass := int64(0)
	inputs := [][]float32{{1, 0}, {3, 44}, {23, 11.3}}
	predictions := [][]float32{
		{-4.14164229, 1.1092185, -0.06021539},
		{10.45007543, -27.46673545, 1.19408663},
		{-5.24206713, 8.45549693, -3.98224414},
	}
	intercepts := []float32{-3.91601811, 0.42575697, 0.13731251}
	predicted_class := []int64{2, 1, 2}
	sg.addAttribute("coefficients", coefficients)
	sg.addAttribute("intercepts", intercepts)
	sg.addAttribute("classlabels_ints", classes)
	sg.addAttribute("multi_class", multiclass)

	sg.addInput("X", []int{3, 2}, inputs)
	sg.addOutput("Y", predicted_class)
	sg.addOutput("Z", predictions)
	sg.errorBound = 0.00001
	sg.Execute(t)
}

func TestLinearClassifierMultiClassProbSigmoid(t *testing.T) {
	sg := Test("LinearClassifier")
	coefficients := []float32{-0.22562418, 0.34188559, 0.68346153,
		-0.68051993, -0.1975279, 0.03748541}
	classes := []int64{1, 2, 3}
	multiclass := int64(0)
	inputs := [][]float32{{1, 0}, {3, 44}, {23, 11.3}}
	predictions := [][]float32{
		{0.015647972, 0.751983387, 0.484950699},
		{0.999971055, 1.17855e-12, 0.767471158},
		{0.005261482, 0.999787317, 0.018302525},
	}
	intercepts := []float32{-3.91601811, 0.42575697, 0.13731251}
	predicted_class := []int64{2, 1, 2}
	sg.addAttribute("coefficients", coefficients)
	sg.addAttribute("intercepts", intercepts)
	sg.addAttribute("classlabels_ints", classes)
	sg.addAttribute("multi_class", multiclass)
	sg.addAttribute("post_transform", []byte("LOGISTIC"))

	sg.addInput("X", []int{3, 2}, inputs)
	sg.addOutput("Y", predicted_class)
	sg.addOutput("Z", predictions)
	sg.errorBound = 0.00001
	sg.Execute(t)
}

func TestLinearClassifierBinary(t *testing.T) {
	sg := Test("LinearClassifier")
	coefficients := []float32{0.00085401, -0.00314063}
	inputs := [][]float32{{1, 0}, {3, 44}, {23, 11.3}}
	predictions := [][]float32{{0.0401599929}, {-0.0963197052}, {0.0234590918}}
	intercepts := []float32{0.03930598}
	predicted_class := []int64{1, 0, 1}
	sg.addAttribute("coefficients", coefficients)
	sg.addAttribute("intercepts", intercepts)

	sg.addInput("X", []int{3, 2}, inputs)
	sg.addOutput("Y", predicted_class)
	sg.addOutput("Z", predictions)
	sg.errorBound = 0.00001
	sg.Execute(t)
}

func TestLinearClassifierBinaryWithLabels(t *testing.T) {
	sg := Test("LinearClassifier")
	coefficients := []float32{0.00085401, -0.00314063}
	inputs := [][]float32{{1, 0}, {3, 44}, {23, 11.3}}
	intercepts := []float32{0.03930598}
	labels := [][]byte{[]byte("not_so_good"), []byte("pretty_good")}
	predictions := [][]float32{{0.959840000, 0.0401599929}, {1.09631968, -0.0963197052}, {0.976540923, 0.0234590918}}
	predicted_class := []string{"pretty_good", "not_so_good", "pretty_good"}
	sg.addAttribute("coefficients", coefficients)
	sg.addAttribute("intercepts", intercepts)
	sg.addAttribute("classlabels_strings", labels)

	sg.addInput("X", []int{3, 2}, inputs)
	sg.addOutput("Y", predicted_class)
	sg.addOutput("Z", predictions)
	sg.errorBound = 0.00001
	sg.Execute(t)
}

func LinearClassifierMultiClass[T ValidInputTypes](t *testing.T) {
	sg := Test("LinearClassifier")
	coefficients := []float32{-0.22562418, 0.34188559, 0.68346153,
		-0.68051993, -0.1975279, 0.03748541}
	classes := []int64{1, 2, 3}
	multiclass := int64(0)
	inputs := [][]T{{1, 0}, {3, 44}, {23, 11}}
	predictions := [][]float32{
		{-4.14164229, 1.1092185, -0.06021539},
		{10.45007543, -27.46673545, 1.19408663},
		{-5.3446321487426758, 8.6596536636352539, -3.9934897422790527},
	}
	intercepts := []float32{-3.91601811, 0.42575697, 0.13731251}
	predicted_class := []int64{2, 1, 2}
	sg.addAttribute("coefficients", coefficients)
	sg.addAttribute("intercepts", intercepts)
	sg.addAttribute("classlabels_ints", classes)
	sg.addAttribute("multi_class", multiclass)

	sg.addInput("X", []int{3, 2}, inputs)
	sg.addOutput("Y", predicted_class)
	sg.addOutput("Z", predictions)
	sg.errorBound = 0.00001
	sg.Execute(t)
}

func TestLinearClassifierMultiClassInt64(t *testing.T) {
	LinearClassifierMultiClass[int64](t)
}

func TestLinearClassifierMultiClassInt32(t *testing.T) {
	LinearClassifierMultiClass[int32](t)
}

func TestLinearClassifierMultiClassDouble(t *testing.T) {
	LinearClassifierMultiClass[float64](t)
}

func TestLinearClassifierTransforms(t *testing.T) {
	values := []struct {
		predicted_class []int64
		predictions     [][]float32
		post_transform  string
	}{
		{
			[]int64{0, 2, 2},
			[][]float32{{2.41, -2.12, 0.59}, {0.67, -1.14, 1.35}, {-1.07, -0.16, 2.11}},
			"NONE",
		},
		{
			[]int64{0, 2, 2},
			[][]float32{{0.917587, 0.107168, 0.643365}, {0.661503, 0.24232, 0.79413}, {0.255403, 0.460085, 0.891871}},
			"LOGISTIC",
		},
		{
			[]int64{0, 2, 2},
			[][]float32{{0.852656, 0.009192, 0.138152}, {0.318722, 0.05216, 0.629118}, {0.036323, 0.090237, 0.87344}},
			"SOFTMAX",
		},
		{
			[]int64{0, 2, 2},
			[][]float32{{0.852656, 0.009192, 0.138152}, {0.318722, 0.05216, 0.629118}, {0.036323, 0.090237, 0.87344}},
			"SOFTMAX_ZERO",
		},
		{
			[]int64{1, 1, 1},
			[][]float32{{-0.527324, -0.445471, -1.080504}, {-0.067731, 0.316014, -0.310748}, {0.377252, 1.405167, 0.295001}},
			"PROBIT",
		},
	}
	inputs := [][]float32{{0, 1}, {2, 3}, {4, 5}}
	for _, v := range values {
		var intercepts []float32
		var coefficients []float32
		if v.post_transform == "PROBIT" {
			coefficients = []float32{0.058, 0.029, 0.09, 0.058, 0.029, 0.09}
			intercepts = []float32{0.27, 0.27, 0.05}
		} else {
			coefficients = []float32{-0.58, -0.29, -0.09, 0.58, 0.29, 0.09}
			intercepts = []float32{2.7, -2.7, 0.5}
		}
		sg := Test("LinearClassifier")
		classes := []int64{0, 1, 2}
		multiclass := int64(0)
		sg.addAttribute("coefficients", coefficients)
		sg.addAttribute("intercepts", intercepts)
		sg.addAttribute("classlabels_ints", classes)
		sg.addAttribute("multi_class", multiclass)
		sg.addAttribute("post_transform", []byte(v.post_transform))
		sg.addInput("X", []int{3, 2}, inputs)
		sg.addOutput("Y", v.predicted_class)
		sg.addOutput("Z", v.predictions)
		sg.errorBound = 0.00001
		sg.Execute(t)
	}
}
