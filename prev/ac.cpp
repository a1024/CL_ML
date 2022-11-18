#include<vector>
#include<string>

	#define		ABAC2_CONF_MSB_RELATION

	#define		MEASURE_PREDICTION

typedef unsigned long long u64;
const double	boost_power=4, min_conf=0.55;
inline int		clamp(int lo, int x, int hi)
{
	if(x<lo)
		x=lo;
	if(x>hi)
		x=hi;
	return x;
}
inline int		load32_big(const unsigned char *data)
{
	return data[0]<<24|data[1]<<16|data[2]<<8|data[3];
}
void			abac2_encode(const void *src, int imsize, int depth, int bytestride, std::string &out_data, int *out_sizes, int *out_conf, bool loud)
{
	auto buffer=(const unsigned char*)src;
	if(!imsize)
		return;
	auto t1=__rdtsc();
	
#ifdef MEASURE_PREDICTION
	u64 hitnum=0, hitden=0;//prediction efficiency
#endif

	std::vector<std::string> planes(depth);
	for(int kp=depth-1;kp>=0;--kp)//bit-plane loop		encode MSB first
	{
		int bit_offset=kp>>3, bit_shift=kp&7;
		int bit_offset2=(kp+1)>>3, bit_shift2=(kp+1)&7;
		auto &plane=planes[depth-1-kp];
		int prob=0x8000, prob_correct=0x8000;//cheap weighted average predictor
#if 1
		u64 hitcount=1;

		for(int kb=0, kb2=0;kb<imsize;++kb, kb2+=bytestride)//analyze bitplane
		{
			int bit=buffer[kb2+bit_offset]>>bit_shift&1;
			int p0=((long long)(prob-0x8000)*prob_correct>>16);
			p0+=0x8000;
			//int p0=0x8000+(long long)(prob-0x8000)*hitcount/(kb+1);
			p0=clamp(1, p0, 0xFFFE);
			int correct=bit^(p0>=0x8000);
			//if(kp==1)
			//	printf("%d", bit);//actual bits
			//	printf("%d", p0<0x8000);//predicted bits
			//	printf("%d", !correct);//prediction error
			hitcount+=correct;
			prob=!bit<<15|prob>>1;
			prob_correct=correct<<15|prob_correct>>1;
		}
		out_conf[depth-1-kp]=(int)hitcount;

		if(hitcount<imsize*min_conf)
		{
			plane.resize((imsize+7)>>3, 0);
			for(int kb=0, kb2=0, b=0;kb<imsize;++kb, kb2+=bytestride)
			{
				int byte_idx=kb>>3, bit_idx=kb&7;
				int bit=buffer[kb2+bit_offset]>>bit_shift&1;
				plane[byte_idx]|=bit<<bit_idx;
			}
			goto done;
		}
		
		int hitratio_sure=int(0x10000*pow((double)hitcount/imsize, 1/boost_power)), hitratio_notsure=int(0x10000*pow((double)hitcount/imsize, boost_power));
		int hitratio_delta=hitratio_sure-hitratio_notsure;
		hitcount=(hitcount<<16)/imsize;

		prob_correct=prob=0x8000;
#endif
#ifdef ABAC2_CONF_MSB_RELATION
		int prevbit0=0;
#endif
		
		plane.reserve(imsize>>8);
		unsigned start=0;
		u64 range=0xFFFFFFFF;
		for(int kb=0, kb2=0;kb<imsize;kb2+=bytestride)//bit-pixel loop		http://mattmahoney.net/dc/dce.html#Section_32
		{
			int bit=buffer[kb2+bit_offset]>>bit_shift&1;
#ifdef ABAC2_CONF_MSB_RELATION
			int prevbit=buffer[kb2+bit_offset2]>>bit_shift2&1;
#endif
			
			if(range<3)
			{
				plane.push_back(start>>24);
				plane.push_back(start>>16&0xFF);
				plane.push_back(start>>8&0xFF);
				plane.push_back(start&0xFF);
				start=0, range=0xFFFFFFFF;//because 1=0.9999...
			}
			
			int p0=prob-0x8000;
			p0=p0*prob_correct>>16;
			p0=p0*prob_correct>>16;
			int sure=-(prevbit==prevbit0);
			p0=p0*(hitratio_notsure+(hitratio_delta&sure))>>16;
			p0+=0x8000;
			p0=clamp(1, p0, 0xFFFE);
			unsigned r2=(unsigned)(range*p0>>16);
			r2+=(r2==0)-(r2==range);
#ifdef DEBUG_ABAC2
			if(kp==examined_plane&&kb>=examined_start&&kb<examined_end)
				printf("%6d %6d %d %08X+%08X %08X %08X\n", kp, kb, bit, start, (int)range, r2, start+r2);
#endif

			int correct=bit^(p0>=0x8000);
			prob=!bit<<15|prob>>1;
			prob_correct=correct<<15|prob_correct>>1;
#ifdef ABAC2_CONF_MSB_RELATION
			prevbit0=prevbit;
#endif
#ifdef MEASURE_PREDICTION
			hitnum+=correct, ++hitden;
#endif
			auto start0=start;
			if(bit)
			{
				++r2;
				start+=r2, range-=r2;
			}
			else
				range=r2-1;
			if(start<start0)//
			{
				printf("OVERFLOW\nstart = %08X -> %08X, r2 = %08X", start0, start, r2);
				int k=0;
				scanf_s("%d", &k);
			}
			++kb;
			
			while((start^(start+(unsigned)range))<0x1000000)//most significant byte has stabilized			zpaq 1.10
			{
#ifdef DEBUG_ABAC2
				if(kp==examined_plane&&kb>=examined_start&&kb<examined_end)
					printf("range %08X byte-out %02X\n", (int)range, start>>24);
#endif
				plane.push_back(start>>24);
				start<<=8;
				range=range<<8|0xFF;
			}
		}
		plane.push_back(start>>24&0xFF);//big-endian
		plane.push_back(start>>16&0xFF);
		plane.push_back(start>>8&0xFF);
		plane.push_back(start&0xFF);
	done:
		if(loud)
			printf("bit %d: conf = %6d / %6d = %lf%%\n", kp, out_conf[depth-1-kp], imsize, 100.*out_conf[depth-1-kp]/imsize);
	}
	auto t_enc=__rdtsc();
	out_data.clear();
	for(int k=0;k<depth;++k)
		out_sizes[k]=(int)planes[k].size();
	for(int k=0;k<depth;++k)
	{
		auto &plane=planes[k];
		out_data.insert(out_data.end(), plane.begin(), plane.end());
	}

	auto t2=__rdtsc();
	if(loud)
	{
		int original_bitsize=imsize*depth, compressed_bitsize=(int)out_data.size()<<3;
		printf("AC encode:  %lld cycles (Enc: %lld cycles)\n", t2-t1, t_enc-t1);
		printf("Size: %d -> %d, ratio: %lf\n", original_bitsize>>3, compressed_bitsize>>3, (double)original_bitsize/compressed_bitsize);
#ifdef MEASURE_PREDICTION
		printf("Predicted: %6lld / %6lld = %lf%%\n", hitnum, hitden, 100.*hitnum/hitden);
#endif
		printf("Bit\tbytes\tratio,\tbytes/bitplane = %d\n", imsize>>3);
		for(int k=0;k<depth;++k)
			printf("%2d\t%5d\t%lf\n", depth-1-k, out_sizes[k], (double)imsize/(out_sizes[k]<<3));
		
		printf("Preview:\n");
		int kprint=(int)(out_data.size()<200?out_data.size():200);
		for(int k=0;k<kprint;++k)
			printf("%02X-", out_data[k]&0xFF);
		printf("\n");
	}
}
void			abac2_decode(const char *data, const int *sizes, const int *conf, void *dst, int imsize, int depth, int bytestride, bool loud)
{
	auto buffer=(unsigned char*)dst;
	if(!imsize)
		return;
	auto t1=__rdtsc();
	memset(buffer, 0, imsize*sizeof(short));
	
	for(int kp=depth-1, cusize=0;kp>=0;--kp)//bit-plane loop
	{
		int bit_offset=kp>>3, bit_shift=kp&7;
		int bit_offset2=(kp+1)>>3, bit_shift2=(kp+1)&7;
		int ncodes=sizes[depth-1-kp];
		auto plane=data+cusize;
		
		int prob=0x8000, prob_correct=0x8000;
#if 1
		u64 hitcount=conf[depth-1-kp];
		if(hitcount<imsize*min_conf)
		{
			for(int kb=0, kb2=0, b=0;kb<imsize;++kb, kb2+=bytestride)
			{
				int byte_idx=kb>>3, bit_idx=kb&7;
				int bit=plane[byte_idx]>>bit_idx&1;
				buffer[kb2+bit_offset]|=bit<<bit_shift;
			}
			cusize+=ncodes;
			continue;
		}
#ifdef ABAC2_CONF_MSB_RELATION
		int prevbit0=0;
#endif
		int hitratio_sure=int(0x10000*pow((double)hitcount/imsize, 1/boost_power)), hitratio_notsure=int(0x10000*pow((double)hitcount/imsize, boost_power));
		int hitratio_delta=hitratio_sure-hitratio_notsure;
		hitcount=(hitcount<<16)/imsize;
#endif

		unsigned start=0;
		u64 range=0xFFFFFFFF;
		unsigned code=load32_big((unsigned char*)plane);
		for(int kc=4, kb=0, kb2=0;kb<imsize;kb2+=bytestride)//bit-pixel loop
		{
			if(range<3)
			{
				code=load32_big((unsigned char*)plane+kc);
				kc+=4;
				start=0, range=0xFFFFFFFF;//because 1=0.9999...
			}
#ifdef ABAC2_CONF_MSB_RELATION
			int prevbit=buffer[kb2+bit_offset2]>>bit_shift2&1;
#endif
			int p0=prob-0x8000;
			p0=p0*prob_correct>>16;
			p0=p0*prob_correct>>16;
			int sure=-(prevbit==prevbit0);
			p0=p0*(hitratio_notsure+(hitratio_delta&sure))>>16;
			p0+=0x8000;
			p0=clamp(1, p0, 0xFFFE);
			unsigned r2=(unsigned)(range*p0>>16);
			r2+=(r2==0)-(r2==range);
			unsigned middle=start+r2;
			int bit=code>middle;
#ifdef DEBUG_ABAC2
			if(kp==examined_plane&&kb>=examined_start&&kb<examined_end)
				printf("%6d %6d %d %08X+%08X %08X %08X %08X\n", kp, kb, bit, start, (int)range, r2, middle, code);
#endif
			
			int correct=bit^(p0>=0x8000);
			prob=!bit<<15|prob>>1;
			prob_correct=correct<<15|prob_correct>>1;
#ifdef ABAC2_CONF_MSB_RELATION
			prevbit0=prevbit;
#endif
			
			if(bit)
			{
				++r2;
				start+=r2, range-=r2;
			}
			else
				range=r2-1;
			
			buffer[kb2+bit_offset]|=bit<<bit_shift;
			++kb;
			
			while((start^(start+(unsigned)range))<0x1000000)//shift-out identical bytes			zpaq 1.10
			{
#ifdef DEBUG_ABAC2
				if(kp==examined_plane&&kb>=examined_start&&kb<examined_end)
					printf("range %08X byte-out %02X\n", (int)range, code>>24);
#endif
				code=code<<8|(unsigned char)plane[kc];
				++kc;
				start<<=8;
				range=range<<8|0xFF;
			}
		}
		cusize+=ncodes;
	}

	auto t2=__rdtsc();
	if(loud)
	{
		printf("AC decode:  %lld cycles\n", t2-t1);
	}
}

int				abac2_estimate(const void *src, int imsize, int depth, int bytestride, int *out_sizes, int *out_conf, bool loud)
{
	auto buffer=(const unsigned char*)src;
	int compr_size=0;
	if(!imsize)
		return compr_size;
	auto t1=__rdtsc();
	
	auto planesizes=new int[depth];
	memset(planesizes, 0, depth*sizeof(int));
#ifdef MEASURE_PREDICTION
	u64 hitnum=0, hitden=0;//prediction efficiency
#endif

	for(int kp=depth-1;kp>=0;--kp)//bit-plane loop		encode MSB first
	{
		int bit_offset=kp>>3, bit_shift=kp&7;
		int bit_offset2=(kp+1)>>3, bit_shift2=(kp+1)&7;
		int prob=0x8000, prob_correct=0x8000;//cheap weighted average predictor
#if 1
		u64 hitcount=1;

		for(int kb=0, kb2=0;kb<imsize;++kb, kb2+=bytestride)//analyze bitplane
		{
			int bit=buffer[kb2+bit_offset]>>bit_shift&1;
			int p0=((long long)(prob-0x8000)*prob_correct>>16);
			p0+=0x8000;
			//int p0=0x8000+(long long)(prob-0x8000)*hitcount/(kb+1);
			p0=clamp(1, p0, 0xFFFE);
			int correct=bit^(p0>=0x8000);
			//if(kp==1)
			//	printf("%d", bit);//actual bits
			//	printf("%d", p0<0x8000);//predicted bits
			//	printf("%d", !correct);//prediction error
			hitcount+=correct;
			prob=!bit<<15|prob>>1;
			prob_correct=correct<<15|prob_correct>>1;
		}
		out_conf[depth-1-kp]=(int)hitcount;

		if(hitcount<imsize*min_conf)
		{
			planesizes[depth-1-kp]=(imsize+7)>>3;
			goto done;
		}
		
		int hitratio_sure=int(0x10000*pow((double)hitcount/imsize, 1/boost_power)), hitratio_notsure=int(0x10000*pow((double)hitcount/imsize, boost_power));
		int hitratio_delta=hitratio_sure-hitratio_notsure;
		hitcount=(hitcount<<16)/imsize;

		prob_correct=prob=0x8000;
#endif
#ifdef ABAC2_CONF_MSB_RELATION
		int prevbit0=0;
#endif
		
		unsigned start=0;
		u64 range=0xFFFFFFFF;
		for(int kb=0, kb2=0;kb<imsize;kb2+=bytestride)//bit-pixel loop		http://mattmahoney.net/dc/dce.html#Section_32
		{
			int bit=buffer[kb2+bit_offset]>>bit_shift&1;
#ifdef ABAC2_CONF_MSB_RELATION
			int prevbit=buffer[kb2+bit_offset2]>>bit_shift2&1;
#endif
			
			if(range<3)
			{
				planesizes[depth-1-kp]+=4;
				start=0, range=0xFFFFFFFF;//because 1=0.9999...
			}
			
			int p0=prob-0x8000;
			p0=p0*prob_correct>>16;
			p0=p0*prob_correct>>16;
			int sure=-(prevbit==prevbit0);
			p0=p0*(hitratio_notsure+(hitratio_delta&sure))>>16;
			p0+=0x8000;
			p0=clamp(1, p0, 0xFFFE);
			unsigned r2=(unsigned)(range*p0>>16);
			r2+=(r2==0)-(r2==range);
#ifdef DEBUG_ABAC2
			if(kp==examined_plane&&kb>=examined_start&&kb<examined_end)
				printf("%6d %6d %d %08X+%08X %08X %08X\n", kp, kb, bit, start, (int)range, r2, start+r2);
#endif

			int correct=bit^(p0>=0x8000);
			prob=!bit<<15|prob>>1;
			prob_correct=correct<<15|prob_correct>>1;
#ifdef ABAC2_CONF_MSB_RELATION
			prevbit0=prevbit;
#endif
#ifdef MEASURE_PREDICTION
			hitnum+=correct, ++hitden;
#endif
			auto start0=start;
			if(bit)
			{
				++r2;
				start+=r2, range-=r2;
			}
			else
				range=r2-1;
			if(start<start0)//
			{
				printf("OVERFLOW\nstart = %08X -> %08X, r2 = %08X", start0, start, r2);
				int k=0;
				scanf_s("%d", &k);
			}
			++kb;
			
			while((start^(start+(unsigned)range))<0x1000000)//most significant byte has stabilized			zpaq 1.10
			{
#ifdef DEBUG_ABAC2
				if(kp==examined_plane&&kb>=examined_start&&kb<examined_end)
					printf("range %08X byte-out %02X\n", (int)range, start>>24);
#endif
				++planesizes[depth-1-kp];
				start<<=8;
				range=range<<8|0xFF;
			}
		}
		planesizes[depth-1-kp]+=4;
	done:
		if(loud)
			printf("bit %d: conf = %6d / %6d = %lf%%\n", kp, out_conf[depth-1-kp], imsize, 100.*out_conf[depth-1-kp]/imsize);
	}
	for(int kp=0;kp<depth;++kp)
		compr_size+=planesizes[depth-1-kp];
	if(out_sizes)
		memcpy(out_sizes, planesizes, depth*sizeof(int));

	auto t2=__rdtsc();
	if(loud)
	{
		int original_bitsize=imsize*depth, compressed_bitsize=(int)compr_size<<3;
		printf("AC estimate:  %lld cycles\n", t2-t1);
		printf("Size: %d -> %d, ratio: %lf\n", original_bitsize>>3, compressed_bitsize>>3, (double)original_bitsize/compressed_bitsize);
#ifdef MEASURE_PREDICTION
		printf("Predicted: %6lld / %6lld = %lf%%\n", hitnum, hitden, 100.*hitnum/hitden);
#endif
		printf("Bit\tbytes\tratio,\tbytes/bitplane = %d\n", imsize>>3);
		for(int k=0;k<depth;++k)
			printf("%2d\t%5d\t%lf\n", depth-1-k, out_sizes[k], (double)imsize/(out_sizes[k]<<3));
		printf("\n");
	}
	delete[] planesizes;
	return compr_size;
}