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

func TestSVMClassifierMulticlassLinearSVC(t *testing.T) {
	sg := Test("SVMClassifier")
	coefficients := []float32{-1.55181212e-01, 2.42698956e-01, 7.01893432e-03,
		4.07614474e-01, -3.24927823e-02, 2.79897536e-04,
		-1.95771302e-01, -3.52437368e-01, -2.15973096e-02,
		-4.38190277e-01, 4.56869105e-02, -1.29375499e-02}
	classes := []int64{0, 1, 2, 3}
	rho := []float32{-0.07489691, -0.1764396, -0.21167431, -0.51619097}
	kernel_params := []float32{0.001, 0, 3}

	input := []float32{1, 0.0, 0.4, 3.0, 44.0, -3,
		12.0, 12.9, -312, 23.0, 11.3, -222,
		23.0, 11.3, -222, 23.0, 3311.3, -222,
		23.0, 11.3, -222, 43.0, 413.3, -114}
	predictions := []int64{1, 0, 1, 1, 1, 0, 1, 0}
	scores := [][]float32{
		{-0.227270544, 0.332829535, -0.279307127, -0.518262208},
		{10.1172562, -0.282575697, -16.1046638, 0.659568906},
		{-0.996162534, 4.30999184, -0.232234091, -0.707304120},
		{-2.45976996, 8.87092972, -3.76557732, -6.76487541},
		{-2.45976996, 8.87092972, -3.76557732, -6.76487541},
		{798.446777, -98.3552551, -1166.80896, 144.001923},
		{-2.45976996, 8.87092972, -3.76557732, -6.76487541},
		{92.7596283, 3.99134970, -151.693329, 1.44020212},
	}
	sg.addAttribute("kernel_type", []byte("LINEAR"))
	sg.addAttribute("coefficients", coefficients)
	sg.addAttribute("rho", rho)
	sg.addAttribute("kernel_params", kernel_params)
	sg.addAttribute("classlabels_ints", classes)

	sg.addInput("X", []int{8, 3}, input)
	sg.addOutput("Y", predictions)
	sg.addOutput("Z", scores)
	sg.errorBound = 0.000001
	sg.isRelativeErr = true
	err := sg.Execute(t)
	if err != nil {
		t.Fatalf("got error: %v", err)
	}
}

func TestSVMClassifierSVCProbabilities(t *testing.T) {
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
	proba := []float32{-3.8214362, 1.82177748, 1.82177748, 7.17655643, 7.17655643, 0.69314718}
	probb := []float32{-1.72839673e+00, -1.12863030e+00, -1.12863030e+00, -6.48340925e+00, -6.48340925e+00, 2.39189538e-16}

	input := []float32{1, 0, 0.4, 3.0, 44.0, -3, 12.0, 12.9, -312, 23.0, 11.3, -222, 23.0, 11.3, -222}
	predictions := []int64{1, 1, 2, 0, 0}
	scores := [][]float32{
		{0.13766955, 0.21030431, 0.32596754, 0.3260586},
		{0.45939931, 0.26975416, 0.13539588, 0.13545066},
		{0.71045899, 0.07858939, 0.05400437, 0.15694726},
		{0.58274772, 0.10203105, 0.15755227, 0.15766896},
		{0.58274772, 0.10203105, 0.15755227, 0.15766896},
	}
	sg.addAttribute("kernel_type", []byte("RBF"))
	sg.addAttribute("coefficients", coefficients)
	sg.addAttribute("support_vectors", support_vectors)
	sg.addAttribute("vectors_per_class", vectors_per_class)
	sg.addAttribute("rho", rho)
	sg.addAttribute("kernel_params", kernel_params)
	sg.addAttribute("classlabels_ints", classes)
	sg.addAttribute("prob_a", proba)
	sg.addAttribute("prob_b", probb)

	sg.addInput("X", []int{5, 3}, input)
	sg.addOutput("Y", predictions)
	sg.addOutput("Z", scores)
	sg.errorBound = 0.000001
	err := sg.Execute(t)
	if err != nil {
		t.Fatalf("got error: %v", err)
	}
}

func TestSVMClassifierSVC(t *testing.T) {
	sg := Test("SVMClassifier")
	coefficients := []float32{1.14360327, 1.95968249, -1.175683, -1.92760275, -1.32575698,
		-1.32575698, 0.66332785, 0.66242913, 0.53120854, 0.53510444,
		-1.06631298, -1.06631298, 0.66332785, 0.66242913, 0.53120854,
		0.53510444, 1, -1,
	}
	support_vectors := []float32{0, 0.5, 32, 2, 2.9, -32, 1, 1.5, 1, 3,
		13.3, -11, 12, 12.9, -312, 43, 413.3, -114,
	}
	classes := []int64{0, 1}
	vectors_per_class := []int64{3, 3}
	rho := []float32{0.5279583}
	kernel_params := []float32{0.001, 0, 3} // gamma, coef0, degree

	input := []float32{1, 0, 0.4, 3.0, 44.0, -3, 12.0, 12.9, -312, 23.0, 11.3, -222, 23.0, 11.3, -222}
	predictions := []int64{1, 1, 1, 0, 0}
	scores := [][]float32{
		{0.95695829391479492, -0.95695829391479492},
		{0.1597825288772583, -0.1597825288772583},
		{0.797798752784729, -0.797798752784729},
		{-0.52760261297225952, 0.52760261297225952},
		{-0.52760261297225952, 0.52760261297225952},
	}
	sg.addAttribute("kernel_type", []byte("RBF"))
	sg.addAttribute("coefficients", coefficients)
	sg.addAttribute("support_vectors", support_vectors)
	sg.addAttribute("vectors_per_class", vectors_per_class)
	sg.addAttribute("rho", rho)
	sg.addAttribute("kernel_params", kernel_params)
	sg.addAttribute("classlabels_ints", classes)

	sg.addInput("X", []int{5, 3}, input)
	sg.addOutput("Y", predictions)
	sg.addOutput("Z", scores)
	sg.errorBound = 0.000001
	err := sg.Execute(t)
	if err != nil {
		t.Fatalf("got error: %v", err)
	}
}

func TestSVMClassifierSVCDouble(t *testing.T) {
	sg := Test("SVMClassifier")
	coefficients := []float32{1.14360327, 1.95968249, -1.175683, -1.92760275, -1.32575698,
		-1.32575698, 0.66332785, 0.66242913, 0.53120854, 0.53510444,
		-1.06631298, -1.06631298, 0.66332785, 0.66242913, 0.53120854,
		0.53510444, 1, -1,
	}
	support_vectors := []float32{0, 0.5, 32, 2, 2.9, -32, 1, 1.5, 1, 3,
		13.3, -11, 12, 12.9, -312, 43, 413.3, -114,
	}
	classes := []int64{0, 1}
	vectors_per_class := []int64{3, 3}
	rho := []float32{0.5279583}
	kernel_params := []float32{0.001, 0, 3} // gamma, coef0, degree

	input := []float64{1, 0, 0.4, 3.0, 44.0, -3, 12.0, 12.9, -312, 23.0, 11.3, -222, 23.0, 11.3, -222}
	predictions := []int64{1, 1, 1, 0, 0}
	scores := [][]float32{
		{0.95695829391479492, -0.95695829391479492},
		{0.1597825288772583, -0.1597825288772583},
		{0.797798752784729, -0.797798752784729},
		{-0.52760261297225952, 0.52760261297225952},
		{-0.52760261297225952, 0.52760261297225952},
	}
	sg.addAttribute("kernel_type", []byte("RBF"))
	sg.addAttribute("coefficients", coefficients)
	sg.addAttribute("support_vectors", support_vectors)
	sg.addAttribute("vectors_per_class", vectors_per_class)
	sg.addAttribute("rho", rho)
	sg.addAttribute("kernel_params", kernel_params)
	sg.addAttribute("classlabels_ints", classes)

	sg.addInput("X", []int{5, 3}, input)
	sg.addOutput("Y", predictions)
	sg.addOutput("Z", scores)
	sg.errorBound = 0.000001
	err := sg.Execute(t)
	if err != nil {
		t.Fatalf("got error: %v", err)
	}
}

func TestSVMClassifierLinear(t *testing.T) {
	sg := Test("SVMClassifier")
	coefficients := []float32{0.766398549079895, 0.0871576070785522, 0.110420741140842, -0.963976919651031}
	support_vectors := []float32{4.80000019073486, 3.40000009536743, 1.89999997615814,
		5, 3, 1.60000002384186,
		4.5, 2.29999995231628, 1.29999995231628,
		5.09999990463257, 2.5, 3,
	}
	classes := []int64{0, 1}
	vectors_per_class := []int64{3, 1}
	rho := []float32{2.23510527610779}
	kernel_params := []float32{0.122462183237076, 0, 3} // gamma, coef0, degree

	input := []float32{5.1, 3.5, 1.4, 4.9, 3, 1.4, 4.7, 3.2, 1.3,
		4.6, 3.1, 1.5, 5, 3.6, 1.4}
	predictions := []int64{0, 0, 0, 0, 0}
	scores := [][]float32{
		{-1.5556798, 1.5556798},
		{-1.2610321, 1.2610321},
		{-1.5795376, 1.5795376},
		{-1.3083477, 1.3083477},
		{-1.6572928, 1.6572928},
	}
	sg.addAttribute("kernel_type", []byte("LINEAR"))
	sg.addAttribute("coefficients", coefficients)
	sg.addAttribute("support_vectors", support_vectors)
	sg.addAttribute("vectors_per_class", vectors_per_class)
	sg.addAttribute("rho", rho)
	sg.addAttribute("kernel_params", kernel_params)
	sg.addAttribute("classlabels_ints", classes)

	sg.addInput("X", []int{5, 3}, input)
	sg.addOutput("Y", predictions)
	sg.addOutput("Z", scores)
	sg.errorBound = 0.000001
	err := sg.Execute(t)
	if err != nil {
		t.Fatalf("got error: %v", err)
	}
}
