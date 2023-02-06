#include<Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include<numpy/arrayobject.h>
#ifdef __GNUC__
#include<x86intrin.h>
#else
#include<intrin.h>
#endif

//	#define	DEBUG_PYENT

//pyentropy declarations
typedef struct ABACResultStruct
{
	unsigned long long
		uncbits,	//uncompressed bit count
		compbits,	//compressed bit count
		predbits,	//predicted bit count
		elapsed;	//elapsed CPU cycles
} ABACResult;
void abac4_testencode(const void *src, size_t symcount, int bitoffset, int bitdepth, size_t bytestride, ABACResult *ret);


//python bindings
static PyObject* testencode(PyObject *self, PyObject *args)//(uint8 image[nsym], nch, bitdepth)
{
#if 1
	PyArrayObject *image;
	int nch, bitdepth, is_signed;
	PyArray_Descr *descr;
	int ndim;
	npy_intp *shape;
	const unsigned char *data;
	ABACResult result={0};
	unsigned char *buf=0;

	if(!PyArg_ParseTuple(args, "Oiii", &image, &nch, &bitdepth, &is_signed))
		return 0;
	descr=PyArray_DESCR(image);
	ndim=PyArray_NDIM(image);
	if(descr->kind!='u'&&descr->kind!='i'||descr->elsize!=1||ndim!=1)
		return Py_BuildValue("");
	shape=PyArray_DIMS(image);
	data=(const unsigned char*)PyArray_DATA(image);

	size_t
		size=*shape,
		symstride=(bitdepth+7)>>3,
		nsym=size/(nch*symstride),
		chstride=nsym*symstride;
	long long ret[4]={0};

	if(is_signed)//move sign bit to LSB
	{
		buf=(unsigned char*)malloc(size);
		for(int k=0;k<size;++k)
		{
			unsigned char val=data[k];
			int neg=val<0;
			val^=-neg;
			val+=neg;
			val<<=1;
			val|=neg;
			buf[k]=val;
		}
		data=buf;
	}

	for(int kc=0;kc<nch;++kc)
	{
		abac4_testencode(data+chstride*kc, nsym, 0, bitdepth, symstride, &result);
		ret[0]+=result.uncbits;
		ret[1]+=result.compbits;
		ret[2]+=result.predbits;
		ret[3]+=result.elapsed;
	}
	if(is_signed)
		free(buf);
#ifdef DEBUG_PYENT
	printf("%016llX %016llX %016llX %016llX\n", ret[0], ret[1], ret[2], ret[3]);
#endif
	return Py_BuildValue("LLLL", ret[0], ret[1], ret[2], ret[3]);
#endif
}
static PyMethodDef entropy_methods[]=
{
	{"testencode", testencode, METH_VARARGS, "Get the exact size of data if compressed with an entropy coder."},
	{0}
};
static PyModuleDef entropy_module=
{
	PyModuleDef_HEAD_INIT,
	"pyentropy",
	"Entropy coder for testing",
	-1,
	entropy_methods,
};
PyMODINIT_FUNC PyInit_pyentropy(void)
{
	//sprintf(0, "LOL");//closes unceremoniously
	return PyModule_Create(&entropy_module);
}


//pyentropy implementation

/*
https://www.nullhardware.com/blog/fixed-point-sine-and-cosine-for-embedded-systems/
Implements the 5-order polynomial approximation to sin(x).
@param i   angle (with 2^15 units/circle)
@return    16 bit fixed point Sine value (4.12) (ie: +4096 = +1 & -4096 = -1)

The result is accurate to within +- 1 count. ie: +/-2.44e-4.
*/
int16_t fpsin(int16_t i)
{
    /* Convert (signed) input to a value between 0 and 8192. (8192 is pi/2, which is the region of the curve fit). */
    /* ------------------------------------------------------------------- */
    i <<= 1;
    uint8_t c = i<0; //set carry for output pos/neg

    if(i == (i|0x4000)) // flip input value to corresponding value in range [0..8192)
        i = (1<<15) - i;
    i = (i & 0x7FFF) >> 1;
    /* ------------------------------------------------------------------- */

    /* The following section implements the formula:
     = y * 2^-n * ( A1 - 2^(q-p)* y * 2^-n * y * 2^-n * [B1 - 2^-r * y * 2^-n * C1 * y]) * 2^(a-q)
    Where the constants are defined as follows:
    */
    enum {A1=3370945099u, B1=2746362156u, C1=292421u};
    enum {n=13, p=32, q=31, r=3, a=12};

    uint32_t y = (C1*((uint32_t)i))>>n;
    y = B1 - (((uint32_t)i*y)>>r);
    y = (uint32_t)i * (y>>n);
    y = (uint32_t)i * (y>>n);
    y = A1 - (y>>(p-q));
    y = (uint32_t)i * (y>>n);
    y = (y+(1UL<<(q-a-1)))>>(q-a); // Rounding

    return c ? -y : y;
}

//Cos(x) = sin(x + pi/2)
#define fpcos(i) fpsin((int16_t)(((uint16_t)(i)) + 8192U))


typedef struct StateStruct
{
	unsigned short hist_zero, hist_correct;
	unsigned start;
	unsigned long long range;
} State;
#define PROB_HALF	0x8000
void abac4_testencode(const void *src, size_t symcount, int bitoffset, int bitdepth, size_t bytestride, ABACResult *ret)
{
	const unsigned char *buffer=(const unsigned char*)src;
	unsigned long long csize=0, hitcount=0;
	long long t1=__rdtsc();

	for(int kp=bitdepth-1;kp>=0;--kp)//bit-plane loop		encode MSB first
	{
		State state={PROB_HALF, PROB_HALF, 0, 0xFFFFFFFF};
		int bit_offset=(bitoffset+kp)>>3, bit_shift=(bitoffset+kp)&7;

		for(size_t ks=0, ks2=0;ks<symcount;ks2+=bytestride)//symbol loop
		{
#ifdef DEBUG_PYENT
			if(!(ks%1000))//
				printf("kp %d ks %d\n", kp, ks);//
#endif
			int bit=buffer[ks2+bit_offset]>>bit_shift&1;

			int p0=state.hist_zero-PROB_HALF;		//calculate probability of zero-bit (p0)
			p0=(int)((long long)p0*state.hist_correct>>16);
			p0=(int)((long long)p0*p0>>16);
			p0+=PROB_HALF;

			if(p0<1)				//clamp p0
				p0=1;
			if(p0>(PROB_HALF<<1)-2)
				p0=(PROB_HALF<<1)-2;

			unsigned r2=(unsigned)(state.range*p0>>16);//calculate new range
			r2+=(r2==0)-(r2==state.range);

			if(bit)					//encode bit
			{
				++r2;
				state.start+=r2;
				state.range-=r2;
			}
			else
				state.range=r2-1;

			int correct=bit^(p0>=PROB_HALF);	//update history
			hitcount+=correct;
			state.hist_zero=!bit<<15|state.hist_zero>>1;
			state.hist_correct=correct<<15|state.hist_correct>>1;

			++ks;
			
			while((state.start^(state.start+(unsigned)state.range))<0x1000000)//renormalize			zpaq 1.10
			{
				++csize;
				//dlist_push_back1(&list, (char*)&start+3);

				state.start<<=8;
				state.range=state.range<<8|0xFF;
			}

			if(state.range<3)			//check for underflow
			{
				csize+=4;
				//dlist_push_back1(&list, (char*)&start+3);//big endian
				//dlist_push_back1(&list, (char*)&start+2);
				//dlist_push_back1(&list, (char*)&start+1);
				//dlist_push_back1(&list, &start);

				state.start=0, state.range=0xFFFFFFFF;//because 1=0.9999...
			}
		}
		csize+=4;
	}
	t1=__rdtsc()-t1;
	if(ret)
	{
		ret->uncbits=symcount*bitdepth;
		ret->compbits=csize<<3;
		ret->predbits=hitcount;
		ret->elapsed=t1;
	}
}
