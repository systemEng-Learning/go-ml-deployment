package tests

import "testing"

func TestSVMRegressorSVC(t *testing.T) {
	sg := Test("SVMRegressor")
	coefficients := []float32{-1.54236563, 0.53485162, -1.5170623, 0.69771864, 1.82685767}
	support_vectors := []float32{0, 0.5, 32, 1., 1.5, 1, 2, 2.9, -32, 12, 12.9, -312, 43, 413.3, -114}
	rho := []float32{1.96292297}
	kernel_params := []float32{0.001, 0, 3} // gamma, coef0, degree
	input := []float32{1, 0.0, 0.4, 3.0, 44.0, -3, 12.0, 12.9, -312, 23.0, 11.3, -222, 23.0,
		11.3, -222, 23.0, 3311.3, -222, 23.0, 11.3, -222, 43.0, 413.3, -114}
	predictions := [][]float32{{1.40283655}, {1.86065906}, {2.66064161}, {1.96311014}, {1.96311014}, {1.96292297}, {1.96311014}, {3.78978065}}
	sg.addAttribute("kernel_type", []byte("RBF"))
	sg.addAttribute("coefficients", coefficients)
	sg.addAttribute("support_vectors", support_vectors)
	sg.addAttribute("rho", rho)
	sg.addAttribute("kernel_params", kernel_params)
	sg.addAttribute("n_supports", int64(5))

	sg.addInput("X", []int{8, 3}, input)
	sg.addOutput("Y", predictions)
	sg.errorBound = 0.00001
	sg.Execute(t)
}
