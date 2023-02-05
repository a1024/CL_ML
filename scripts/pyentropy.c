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
		compbits,	//compressed bit count
		predbits,	//predicted bit count
		uncbits,	//uncompressed bit count
		elapsed;	//elapsed CPU cycles
} ABACResult;
void abac4_testencode(const void *src, int symcount, int bitoffset, int bitdepth, int bytestride, ABACResult *ret);


//python bindings
static PyObject* testencode(PyObject *self, PyObject *args)//(uint8 image[nsym], nch, bitdepth)
{
#if 1
	PyArrayObject *image;
	int nch, bitdepth;
	PyArray_Descr *descr;
	int ndim;
	npy_intp *shape;
	const unsigned char *data;
	ABACResult result={0};

	if(!PyArg_ParseTuple(args, "Oii", &image, &nch, &bitdepth))
		return 0;
	descr=PyArray_DESCR(image);
	ndim=PyArray_NDIM(image);
	if(descr->kind!='u'&&descr->kind!='i'||descr->elsize!=1||ndim!=1)
		return Py_BuildValue("");
	shape=PyArray_DIMS(image);
	data=(const unsigned char*)PyArray_DATA(image);

	int nsym=(int)(*shape/nch),
		symstride=(bitdepth+7)>>3,
		chstride=nsym*symstride;
	long long ret[4]={0};

	for(int kc=0;kc<nch;++kc)
	{
#ifdef DEBUG_PYENT
		printf("CHANNEL %d\n", kc);//
#endif
		abac4_testencode(data+chstride*kc, nsym, 0, bitdepth, symstride, &result);
		ret[0]+=result.uncbits;
		ret[1]+=result.compbits;
		ret[2]+=result.predbits;
		ret[3]+=result.elapsed;
	}
#ifdef DEBUG_PYENT
	printf("%016llX %016llX %016llX %016llX\n", ret[0], ret[1], ret[2], ret[3]);
#endif
	return Py_BuildValue("LLLL", ret[0], ret[1], ret[2], ret[3]);
#endif
#if 0
//pyentropy.testencode(3D (u)int8 numpy array, int bitdepth, bool interleaved_channels)
//interleaved_channels: true for [H, W, C], false for [C, H, W]
	PyArrayObject *image;
	PyArray_Descr *descr;
	int ndim;
	npy_intp *shape;
	const unsigned char *data;
	int bitdepth=0, interleaved=0;
	ABACResult result={0};

	if(!PyArg_ParseTuple(args, "Oii", &image, &bitdepth, &interleaved))
		return 0;
	descr=PyArray_DESCR(image);
	ndim=PyArray_NDIM(image);
	if(descr->kind!='u'&&descr->kind!='i'||descr->elsize!=1||ndim!=3||(interleaved&~1))
		return Py_BuildValue("");
	shape=PyArray_DIMS(image);
	data=(const unsigned char*)PyArray_DATA(image);

	npy_intp iw=shape[!interleaved+1], ih=shape[!interleaved], nch=shape[interleaved<<1],

		npixels=iw*ih,
		*strides=PyArray_STRIDES(image),
		//bytestride=(nch*bitdepth+7)>>3,
		chstride=strides[interleaved<<1];

	long long ret[4]={0};

	for(int kc=0;kc<nch;++kc)
	{
		abac4_testencode(data+chstride, (int)npixels, 0, bitdepth, (int)strides[2-interleaved], &result);
		ret[0]+=result.uncbits;
		ret[1]+=result.compbits;
		ret[2]+=result.predbits;
		ret[3]+=result.elapsed;
		break;//
	}

	return Py_BuildValue("LLLL", ret, ret+1, ret+2, ret+3);
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
typedef struct StateStruct
{
	unsigned short prob, prob_correct;
	unsigned start;
	unsigned long long range;
} State;
#define PROB_HALF	0x8000
void abac4_testencode(const void *src, int symcount, int bitoffset, int bitdepth, int bytestride, ABACResult *ret)
{
	const unsigned char *buffer=(const unsigned char*)src;
	unsigned long long csize=0, hitcount=0;
	long long t1=__rdtsc();

	for(int kp=bitdepth-1;kp>=0;--kp)//bit-plane loop		encode MSB first
	{
		State state={PROB_HALF, PROB_HALF, 0, 0xFFFFFFFF};
		int bit_offset=(bitoffset+kp)>>3, bit_shift=(bitoffset+kp)&7;

		for(int ks=0, ks2=0;ks<symcount;ks2+=bytestride)//symbol loop
		{
#ifdef DEBUG_PYENT
			if(!(ks%1000))//
				printf("kp %d ks %d\n", kp, ks);//
#endif
			int bit=buffer[ks2+bit_offset]>>bit_shift&1;

			int p0=state.prob-PROB_HALF;		//calculate probability of zero-bit (p0)
			p0=(int)((long long)p0*state.prob_correct>>16);
			p0=(int)((long long)p0*state.prob_correct>>16);
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

			int correct=bit^(p0>=PROB_HALF);	//update probabilities
			state.prob=!bit<<15|state.prob>>1;
			state.prob_correct=correct<<15|state.prob_correct>>1;

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
		ret->compbits=csize<<3;
		ret->predbits=hitcount;
		ret->uncbits=symcount*bitdepth;
		ret->elapsed=t1;
	}
}
