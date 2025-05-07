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
	predictions := [][]float32{{1.40283655}, {1.86065906}, {2.66064161}, {1.96311014},
		{1.96311014}, {1.96292297}, {1.96311014}, {3.78978065}}
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

func TestSVMRegressorNuSVC(t *testing.T) {
	sg := Test("SVMRegressor")
	coefficients := []float32{-1.7902966, 1.05962596, -1.54324389, -0.43658884, 0.79025169, 1.92025169}
	support_vectors := []float32{0, 0.5, 32, 1, 1.5, 1, 2, 2.9, -32, 3, 13.3, -11, 12, 12.9, -312, 43, 413.3, -114}
	rho := []float32{1.96923464}
	kernel_params := []float32{0.001, 0, 3} // gamma, coef0, degree
	input := []float32{1, 0.0, 0.4, 3.0, 44.0, -3, 12.0, 12.9, -312, 23.0, 11.3, -222, 23.0,
		11.3, -222, 23.0, 3311.3, -222, 23.0, 11.3, -222, 43.0, 413.3, -114}
	predictions := [][]float32{{1.51230766}, {1.77893206}, {2.75948633}, {1.96944663}, {1.96944663},
		{1.96923464}, {1.96944663}, {3.88948633}}
	sg.addAttribute("kernel_type", []byte("RBF"))
	sg.addAttribute("coefficients", coefficients)
	sg.addAttribute("support_vectors", support_vectors)
	sg.addAttribute("rho", rho)
	sg.addAttribute("kernel_params", kernel_params)
	sg.addAttribute("n_supports", int64(6))

	sg.addInput("X", []int{8, 3}, input)
	sg.addOutput("Y", predictions)
	sg.errorBound = 0.00001
	sg.Execute(t)
}

func TestSVMRegressorNuSVCPoly(t *testing.T) {
	sg := Test("SVMRegressor")
	coefficients := []float32{-2.74322388e+01, 5.81893108e+01, -1.00000000e+02,
		6.91693781e+01, 7.62161261e-02, -2.66618042e-03}
	support_vectors := []float32{0, 0.5, 32, 1, 1.5, 1, 2, 2.9, -32, 3, 13.3, -11, 12, 12.9, -312, 43, 413.3, -114}
	rho := []float32{1.5004596}
	kernel_params := []float32{0.001, 0, 3} // gamma, coef0, degree
	input := []float32{1, 0.0, 0.4, 3.0, 44.0, -3, 12.0, 12.9, 112, 23.0, 11.3, -222, 23.0,
		11.3, -222, 23.0, 3311.3, -222, 23.0, 11.3, -222, 43.0, 413.3, -114}
	predictions := [][]float32{{1.50041862e+00}, {3.49624789e-01},
		{-1.36680453e+02}, {-2.28659315e+02},
		{-2.28659315e+02}, {-6.09640827e+05},
		{-2.28659315e+02}, {3.89055458e+00},
	}
	sg.addAttribute("kernel_type", []byte("POLY"))
	sg.addAttribute("coefficients", coefficients)
	sg.addAttribute("support_vectors", support_vectors)
	sg.addAttribute("rho", rho)
	sg.addAttribute("kernel_params", kernel_params)
	sg.addAttribute("n_supports", int64(6))

	sg.addInput("X", []int{8, 3}, input)
	sg.addOutput("Y", predictions)
	sg.errorBound = 0.01
	sg.isRelativeErr = true
	sg.Execute(t)
}

func TestSVMRegressorLinear(t *testing.T) {
	sg := Test("SVMRegressor")
	coefficients := []float32{0.28290501, -0.0266512, 0.01674867}
	rho := []float32{1.24032312}
	kernel_params := []float32{0.001, 0, 3} // gamma, coef0, degree
	input := []float32{1, 0.0, 0.4, 3.0, 44.0, -3, 12.0, 12.9, -312, 23.0, 11.3, -222, 23.0,
		11.3, -222, 23.0, 3311.3, -222, 23.0, 11.3, -222, 43.0, 413.3, -114}
	predictions := [][]float32{{1.52992759}, {0.8661395},
		{-0.93420165}, {3.72777548},
		{3.72777548}, {-84.22117216},
		{3.72777548}, {0.48095091},
	}
	sg.addAttribute("kernel_type", []byte("LINEAR"))
	sg.addAttribute("coefficients", coefficients)
	sg.addAttribute("rho", rho)
	sg.addAttribute("kernel_params", kernel_params)
	sg.addAttribute("n_supports", int64(0))

	sg.addInput("X", []int{8, 3}, input)
	sg.addOutput("Y", predictions)
	sg.errorBound = 0.0025
	sg.Execute(t)
}
