#ifdef CLFUNC

CLFUNC(zeromem)
CLFUNC(relu)
CLFUNC(res_add)
CLFUNC(conv11)
CLFUNC(conv_zp)
CLFUNC(maxpool)
CLFUNC(avpool)
CLFUNC(linear_fc)

#if 0
CLFUNC(pad_data)
//CLFUNC(res_save)//use clEnqueueCopyBuffer instead
CLFUNC(res_save_ds)
CLFUNC(res_add)
CLFUNC(relu)
CLFUNC(maxpool_n22_zp)
CLFUNC(avpool_n77)
//CLFUNC(batchnorm)//pre-baked into conv
CLFUNC(conv_n11)
CLFUNC(conv_n33_zp)
CLFUNC(conv_n77_zp)
#endif

#if 0
CLFUNC(add)
CLFUNC(relu)
CLFUNC(relu_inplace)
CLFUNC(lrelu_inplace)
CLFUNC(conv33_const)
CLFUNC(conv33_mirror)
CLFUNC(conv33_periodic)

CLFUNC(lift_2D_H_ohi)
CLFUNC(lift_2D_H_elo)
CLFUNC(lift_2D_V_ohi)
CLFUNC(lift_2D_V_elo)

CLFUNC(permute_even_odd)
CLFUNC(permute_even_odd_inv)
CLFUNC(quantize)
CLFUNC(dequantize)
CLFUNC(quantize_smooth)
CLFUNC(dequantize_smooth)
CLFUNC(sq_error)
#endif

#endif
