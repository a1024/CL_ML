#include"array.h"
#include"error.h"
#include"util.h"
#include<stdlib.h>
#include<string.h>
static const char file[]=__FILE__;

//ARRAY
#if 1
static void		array_realloc(ArrayHandle *arr, size_t count, size_t pad)//CANNOT be nullptr, array must be initialized with array_alloc()
{
	ArrayHandle p2;
	size_t size, newcap;

	ASSERT_MSG(*arr, "array_realloc(): ArrayHandle is NULL");
	size=(count+pad)*arr[0]->esize, newcap=arr[0]->esize;
	for(;newcap<size;newcap<<=1);
	if(newcap>arr[0]->cap)
	{
		p2=(ArrayHandle)realloc(*arr, sizeof(ArrayHeader)+newcap);
		ASSERT_MSG(p2, "array_realloc(): realloc returned NULL");
		*arr=p2;
		if(arr[0]->cap<newcap)
			memset(arr[0]->data+arr[0]->cap, 0, newcap-arr[0]->cap);
		arr[0]->cap=newcap;
	}
	arr[0]->count=count;
}

//Array API
ArrayHandle		array_construct(const void *src, size_t esize, size_t count, size_t rep, size_t pad, void (*destructor)(void*))
{
	ArrayHandle arr;
	size_t srcsize, dstsize, cap;
	
	srcsize=count*esize;
	dstsize=rep*srcsize;
	cap=dstsize+pad*esize;
	arr=(ArrayHandle)malloc(sizeof(ArrayHeader)+cap);
	ASSERT_MSG(arr, "array_construct(): malloc returned NULL");
	arr->count=count;
	arr->esize=esize;
	arr->cap=cap;
	arr->destructor=destructor;
	if(src)
	{
		ASSERT_PTR(src);
		memfill(arr->data, src, dstsize, srcsize);
	}
	else
		memset(arr->data, 0, dstsize);
		
	if(cap-dstsize>0)//zero pad
		memset(arr->data+dstsize, 0, cap-dstsize);
	return arr;
}
ArrayHandle		array_copy(ArrayHandle *arr)
{
	ArrayHandle a2;
	size_t bytesize;

	if(!*arr)
		return 0;
	bytesize=sizeof(ArrayHeader)+arr[0]->cap;
	a2=(ArrayHandle)malloc(bytesize);
	ASSERT_MSG(a2, "array_copy(): mallco returned NULL");
	memcpy(a2, *arr, bytesize);
	return a2;
}
void			array_clear(ArrayHandle *arr)//can be nullptr
{
	if(*arr)
	{
		if(arr[0]->destructor)
		{
			for(size_t k=0;k<arr[0]->count;++k)
				arr[0]->destructor(array_at(arr, k));
		}
		arr[0]->count=0;
	}
}
void			array_free(ArrayHandle *arr)//can be nullptr
{
	if(*arr&&arr[0]->destructor)
	{
		for(size_t k=0;k<arr[0]->count;++k)
			arr[0]->destructor(array_at(arr, k));
	}
	free(*arr);
	*arr=0;
}
void			array_fit(ArrayHandle *arr, size_t pad)//can be nullptr
{
	ArrayHandle p2;
	if(!*arr)
		return;
	arr[0]->cap=(arr[0]->count+pad)*arr[0]->esize;
	p2=(ArrayHandle)realloc(*arr, sizeof(ArrayHeader)+arr[0]->cap);
	ASSERT_MSG(p2, "array_fit(): realloc returned NULL");
	*arr=p2;
}

void*			array_insert(ArrayHandle *arr, size_t idx, const void *data, size_t count, size_t rep, size_t pad)//cannot be nullptr
{
	size_t start, srcsize, dstsize, movesize;
	
	ASSERT_PTR(*arr);
	start=idx*arr[0]->esize;
	srcsize=count*arr[0]->esize;
	dstsize=rep*srcsize;
	movesize=arr[0]->count*arr[0]->esize-start;
	array_realloc(arr, arr[0]->count+rep*count, pad);
	memmove(arr[0]->data+start+dstsize, arr[0]->data+start, movesize);
	if(data)
		memfill(arr[0]->data+start, data, dstsize, srcsize);
	else
		memset(arr[0]->data+start, 0, dstsize);
	return arr[0]->data+start;
}
void*			array_erase(ArrayHandle *arr, size_t idx, size_t count)
{
	size_t k;

	ASSERT_PTR(*arr);
	if(arr[0]->count<idx+count)
	{
		LOG_ERROR("array_erase() out of bounds: idx=%lld count=%lld size=%lld", (long long)idx, (long long)count, (long long)arr[0]->count);
		if(arr[0]->count<idx)
			return 0;
		count=arr[0]->count-idx;//erase till end of array if just idx+count is OOB
	}
	if(arr[0]->destructor)
	{
		for(k=0;k<count;++k)
			arr[0]->destructor(array_at(arr, idx+k));
	}
	memmove(arr[0]->data+idx*arr[0]->esize, arr[0]->data+(idx+count)*arr[0]->esize, (arr[0]->count-(idx+count))*arr[0]->esize);
	arr[0]->count-=count;
	return arr[0]->data+idx*arr[0]->esize;
}
void*			array_replace(ArrayHandle *arr, size_t idx, size_t rem_count, const void *data, size_t ins_count, size_t rep, size_t pad)
{
	size_t k, c0, c1, start, srcsize, dstsize;

	ASSERT_PTR(*arr);
	if(arr[0]->count<idx+rem_count)
	{
		LOG_ERROR("array_replace() out of bounds: idx=%lld rem_count=%lld size=%lld ins_count=%lld", (long long)idx, (long long)rem_count, (long long)arr[0]->count, (long long)ins_count);
		if(arr[0]->count<idx)
			return 0;
		rem_count=arr[0]->count-idx;//erase till end of array if just idx+count is OOB
	}
	if(arr[0]->destructor)
	{
		for(k=0;k<rem_count;++k)//destroy removed objects
			arr[0]->destructor(array_at(arr, idx+k));
	}
	start=idx*arr[0]->esize;
	srcsize=ins_count*arr[0]->esize;
	dstsize=rep*srcsize;
	c0=arr[0]->count;//copy original count
	c1=arr[0]->count+rep*ins_count-rem_count;//calculate new count

	if(ins_count!=rem_count||(c1+pad)*arr[0]->esize>arr[0]->cap)//resize array
		array_realloc(arr, c1, pad);

	if(ins_count!=rem_count)//shift objects
		memmove(arr[0]->data+(idx+ins_count)*arr[0]->esize, arr[0]->data+(idx+rem_count)*arr[0]->esize, (c0-(idx+rem_count))*arr[0]->esize);

	if(dstsize)//initialize inserted range
	{
		if(data)
			memfill(arr[0]->data+start, data, dstsize, srcsize);
		else
			memset(arr[0]->data+start, 0, dstsize);
	}
	return arr[0]->data+start;//return start of inserted range
}

size_t			array_size(ArrayHandle const *arr)//can be nullptr
{
	if(!arr[0])
		return 0;
	return arr[0]->count;
}
void*			array_at(ArrayHandle *arr, size_t idx)
{
	if(!arr[0])
		return 0;
	if(idx>=arr[0]->count)
		return 0;
	return arr[0]->data+idx*arr[0]->esize;
}
void*			array_back(ArrayHandle *arr)
{
	if(!*arr||!arr[0]->count)
		return 0;
	return arr[0]->data+(arr[0]->count-1)*arr[0]->esize;
}
#endif


//double-linked array LIST
#if 1
void			dlist_init(DListHandle list, size_t objsize, size_t objpernode, void (*destructor)(void*))
{
	list->i=list->f=0;
	list->objsize=objsize;
	list->objpernode=objpernode;
	list->nnodes=list->nobj=0;//empty
	list->destructor=destructor;
}
#define			DLIST_COPY_NODE(DST, PREV, NEXT, SRC, PAYLOADSIZE)\
	DST=(DNodeHandle)malloc(sizeof(DNodeHeader)+(PAYLOADSIZE)),\
	DST->prev=PREV,\
	DST->next=NEXT,\
	memcpy(DST->data, SRC->data, PAYLOADSIZE)
void			dlist_copy(DListHandle dst, DListHandle src)
{
	DNodeHandle it;
	size_t payloadsize;

	dlist_init(dst, src->objsize, src->objpernode, src->destructor);
	it=dst->i;
	if(it)
	{
		payloadsize=src->objpernode*src->objsize;

		DLIST_COPY_NODE(dst->f, 0, 0, it, payloadsize);
		//dst->f=(DNodeHandle)malloc(sizeof(DNodeHeader)+payloadsize);
		//memcpy(dst->f->data, it->data, payloadsize);

		dst->i=dst->f;

		it=it->next;

		for(;it;it=it->next)
		{
			DLIST_COPY_NODE(dst->f->next, dst->f, 0, it, payloadsize);
			dst->f=dst->f->next;
		}
	}
	dst->nnodes=src->nnodes;
	dst->nobj=src->nobj;
}
void			dlist_clear(DListHandle list)
{
	DNodeHandle it;

	it=list->i;
	if(it)
	{
		while(it->next)
		{
			if(list->destructor)
			{
				for(size_t k=0;k<list->objpernode;++k)
					list->destructor(it->data+k*list->objsize);
				list->nobj-=list->objpernode;
			}
			it=it->next;
			free(it->prev);
		}
		if(list->destructor)
		{
			for(size_t k=0;k<list->nobj;++k)
				list->destructor(it->data+k*list->objsize);
		}
		free(it);
		list->i=list->f=0;
		list->nobj=list->nnodes=0;
	}
}
void			dlist_appendtoarray(DListHandle list, ArrayHandle *dst)
{
	DNodeHandle it;
	size_t payloadsize;

	if(!*dst)
		*dst=array_construct(0, list->objsize, 0, 0, list->nnodes*list->objpernode, list->destructor);
	else
	{
		if(dst[0]->esize!=list->objsize)
		{
			LOG_ERROR("dlist_appendtoarray(): dst->esize=%d, list->objsize=%d", dst[0]->esize, list->objsize);
			return;
		}
		ARRAY_APPEND(*dst, 0, 0, 0, list->nnodes*list->objpernode);
	}
	it=list->i;
	payloadsize=list->objpernode*list->objsize;
	for(size_t offset=dst[0]->count;it;)
	{
		memcpy(dst[0]->data+offset*list->objsize, it->data, payloadsize);
		offset+=list->objpernode;
		it=it->next;
	}
	dst[0]->count+=list->nobj;
}

static void		dlist_append_node(DListHandle list)
{
	DNodeHandle temp;

	temp=(DNodeHandle)malloc(sizeof(DNodeHeader)+list->objpernode*list->objsize);
	ASSERT_MSG(temp, "dlist_append_node(): malloc returned NULL");
	temp->next=0;
	if(list->nnodes)
	{
		temp->prev=list->f;
		list->f->next=temp;
	}
	else
	{
		temp->prev=0;
		list->i=temp;
	}
	list->f=temp;
	++list->nnodes;
}
void*			dlist_push_back1(DListHandle list, const void *obj)
{
	size_t obj_idx=list->nobj%list->objpernode;//index of next object
	if(!obj_idx)//need a new node
		dlist_append_node(list);
	void *p=list->f->data+obj_idx*list->objsize;
	if(obj)
		memcpy(p, obj, list->objsize);
	else
		memset(p, 0, list->objsize);
	++list->nobj;
	return p;
}
#if 0
#define			dlist_fill_node(LIST, COPYSIZE, SRC, DST)\
	LIST->nobj+=COPYSIZE;\
	COPYSIZE*=LIST->objsize;\
	if(SRC)\
		memcpy(DST, SRC, COPYSIZE), SRC+=COPYSIZE;\
	else\
		memset(DST, 0, COPYSIZE);
#else
static void		dlist_fill_node(DListHandle list, size_t copysize, char **src, void *dst)
{
	list->nobj+=copysize;
	copysize*=list->objsize;

	if(*src)
		memcpy(dst, *src, copysize), *src+=copysize;
	else
		memset(dst, 0, copysize);
}
#endif
void*			dlist_push_back(DListHandle list, const void *data, size_t count)
{
	size_t obj_idx, copysize;
	char *buffer;
	void *ret;
	
	buffer=(char*)data;
	ret=0;
	obj_idx=list->nobj%list->objpernode;
	if(obj_idx)
	{
		copysize=list->objpernode<obj_idx+count?list->objpernode-obj_idx:count;
		count-=copysize;
		ret=list->f->data+obj_idx*list->objsize;

		dlist_fill_node(list, copysize, &buffer, ret);
	}
	while(count)
	{
		dlist_append_node(list);
		
		copysize=list->objpernode<count?list->objpernode:count;
		count-=copysize;

		if(!ret)
			ret=list->f->data;
		dlist_fill_node(list, copysize, &buffer, list->f->data);
	}
	return ret;
}
void*			dlist_back(DListHandle list)
{
	size_t obj_idx;

	if(!list->nobj)
	{
		LOG_ERROR("dlist_back() called on empty list");
		return 0;
	}
	obj_idx=(list->nobj-1)%list->objpernode;
	return list->f->data+obj_idx*list->objsize;
}
void			dlist_pop_back(DListHandle list)
{
	size_t obj_idx;

	if(!list->nobj)
		LOG_ERROR("dlist_pop_back() called on empty list");
	if(list->destructor)
		list->destructor(dlist_back(list));
	obj_idx=(list->nobj-1)%list->objpernode;
	if(!obj_idx)//last object is first in the last block
	{
		DNodeHandle last=list->f;
		list->f=list->f->prev;
		free(last);
		--list->nnodes;
		if(!list->nnodes)//last object was popped out
			list->i=0;
	}
	--list->nobj;
}

void			dlist_first(DListHandle list, DListItHandle it)
{
	it->list=list;
	it->node=list->i;
	it->obj_idx=0;
}
void			dlist_last(DListHandle list, DListItHandle it)
{
	it->list=list;
	it->node=list->f;
	it->obj_idx=(list->nobj-1)%list->objpernode;
}
void*			dlist_it_deref(DListItHandle it)
{
	if(it->obj_idx>=it->list->nobj)
		LOG_ERROR("dlist_it_deref() out of bounds");
	if(!it->node)
		LOG_ERROR("dlist_it_deref() node == nullptr");
	return it->node->data+it->obj_idx%it->list->objpernode*it->list->objsize;
}
int				dlist_it_inc(DListItHandle it)
{
	++it->obj_idx;
	if(it->obj_idx>=it->list->objpernode)
	{
		it->obj_idx=0;
		if(!it->node||!it->node->next)
			return 0;
			//LOG_ERROR("dlist_it_inc() attempting to read node == nullptr");
		it->node=it->node->next;
	}
	return 1;
}
int				dlist_it_dec(DListItHandle it)
{
	if(it->obj_idx)
		--it->obj_idx;
	else
	{
		if(!it->node||!it->node->prev)
			return 0;
			//LOG_ERROR("dlist_it_dec() attempting to read node == nullptr");
		it->node=it->node->prev;
		it->obj_idx=it->list->objpernode-1;
	}
	return 1;
}
#endif
