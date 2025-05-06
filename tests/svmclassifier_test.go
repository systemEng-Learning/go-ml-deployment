package tests

import "testing"

func TestSVMClassifierMulticlassSVC(t *testing.T) {
	sg := Test("SVMClassifier")
	coefficients := []float32{1.14360327, 1.95968249, -1.175683, -1.92760275, -1.32575698,
		-1.32575698, 0.66332785, 0.66242913, 0.53120854, 0.53510444,
		-1.06631298, -1.06631298, 0.66332785, 0.66242913, 0.53120854,
		0.53510444, 1, -1,
	}
	support_vectors := []float32{0, 0.5, 32, 2, 2.9, -32, 1, 1.5, 1, 3,
		13.3, -11, 12, 12.9, -312, 43, 413.3, -114,
	}
	classes := []int64{0, 1, 2, 3}
	vectors_per_class := []int64{2, 2, 1, 1}
	rho := []float32{0.5279583, 0.32605162, 0.32605162, 0.06663721, 0.06663721, 0}
	kernel_params := []float32{0.001, 0, 3} // gamma, coef0, degree

	input := []float32{1, 0, 0.4, 3.0, 44.0, -3, 12.0, 12.9, -312, 23.0,
		11.3, -222, 23.0, 11.3, -222, 23.0, 3311.3, -222, 23.0,
		11.3, -222, 43.0, 413.3, -114,
	}
	predictions := []int64{1, 1, 2, 0, 0, 0, 0, 3}
	scores := [][]float32{
		{-0.956958294, 0.799815655, 0.799815655, 0.988598406, 0.988598406, 0},
		{-0.159782529, 0.407864451, 0.407864451, 0.347750872, 0.347750872, 0},
		{0.527958274, -0.999705434, 0.326051623, -0.999675810, 0.0666372105, 1.00000000},
		{0.527958274, 0.325695992, 0.326051623, 0.0663511604, 0.0666372105, 0.000268258271},
		{0.527958274, 0.325695992, 0.326051623, 0.0663511604, 0.0666372105, 0.000268258271},
		{0.527958274, 0.326051623, 0.326051623, 0.0666372105, 0.0666372105, 0},
		{0.527958274, 0.325695992, 0.326051623, 0.0663511604, 0.0666372105, 0.000268258271},
		{0.527958274, 0.326051623, -0.999705434, 0.0666372105, -0.999675810, -1.00000000},
	}
	sg.addAttribute("kernel_type", []byte("RBF"))
	sg.addAttribute("coefficients", coefficients)
	sg.addAttribute("support_vectors", support_vectors)
	sg.addAttribute("vectors_per_class", vectors_per_class)
	sg.addAttribute("rho", rho)
	sg.addAttribute("kernel_params", kernel_params)
	sg.addAttribute("classlabels_ints", classes)

	sg.addInput("X", []int{8, 3}, input)
	sg.addOutput("Y", predictions)
	sg.addOutput("Z", scores)
	sg.errorBound = 0.000001
	err := sg.Execute(t)
	if err != nil {
		t.Fatalf("got error: %v", err)
	}
}
