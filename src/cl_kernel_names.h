#ifdef CLKERNEL

CLKERNEL(cc2d)
CLKERNEL(cc2d_grad_in)
CLKERNEL(cc2d_grad_filt)
CLKERNEL(cc2d_grad_bias)

CLKERNEL(lrelu)
CLKERNEL(lrelu_grad)
CLKERNEL(relu)
CLKERNEL(relu_grad)

CLKERNEL(quantizer_train)
CLKERNEL(quantizer_test)
CLKERNEL(loss_MSE)

CLKERNEL(opt_adam)

#endif