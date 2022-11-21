#pragma once
#ifndef INC_AWM_ARRAY_H
#define INC_AWM_ARRAY_H
#include<stddef.h>//for size_t
#ifdef __cplusplus
extern "C"
{
#endif

//ARRAY
#if 1
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4200)//no default-constructor for struct with zero-length array
#endif
typedef struct ArrayHeaderStruct
{
	size_t count, esize, cap;//cap is in bytes
	void (*destructor)(void*);
	unsigned char data[];
} ArrayHeader, *ArrayHandle;
//typedef const ArrayHeader *ArrayConstHandle;
#ifdef _MSC_VER
#pragma warning(pop)
#endif
void			array_realloc(ArrayHandle *arr, size_t count, size_t pad);
ArrayHandle		array_construct(const void *src, size_t esize, size_t count, size_t rep, size_t pad, void (*destructor)(void*));
ArrayHandle		array_copy(ArrayHandle *arr);//shallow
void			array_clear(ArrayHandle *arr);//keeps allocation
void			array_free(ArrayHandle *arr);
void			array_fit(ArrayHandle *arr, size_t pad);

void*			array_insert(ArrayHandle *arr, size_t idx, const void *data, size_t count, size_t rep, size_t pad);//cannot be nullptr
void*			array_erase(ArrayHandle *arr, size_t idx, size_t count);
void*			array_replace(ArrayHandle *arr, size_t idx, size_t rem_count, const void *data, size_t ins_count, size_t rep, size_t pad);

void*			array_at(ArrayHandle *arr, size_t idx);
void*			array_back(ArrayHandle *arr);

#define			ARRAY_ALLOC(ELEM_TYPE, ARR, DATA, COUNT, PAD, DESTRUCTOR)	ARR=array_construct(DATA, sizeof(ELEM_TYPE), COUNT, 1, PAD, DESTRUCTOR)
#define			ARRAY_APPEND(ARR, DATA, COUNT, REP, PAD)					array_insert(&(ARR), (ARR)->count, DATA, COUNT, REP, PAD)
#define			ARRAY_DATA(ARR)			(ARR)->data
#define			ARRAY_I(ARR, IDX)		*(int*)array_at(&ARR, IDX)
#define			ARRAY_U(ARR, IDX)		*(unsigned*)array_at(&ARR, IDX)
#define			ARRAY_F(ARR, IDX)		*(double*)array_at(&ARR, IDX)


//null terminated array
#define			ESTR_ALLOC(TYPE, STR, DATA, LEN)	STR=array_construct(DATA, sizeof(TYPE), LEN, 1, 1, 0)
#define			STR_APPEND(STR, SRC, LEN, REP)		array_insert(&(STR), (STR)->count, SRC, LEN, REP, 1)
#define			STR_POPBACK(STR, COUNT)				array_erase(&(STR), (STR)->count-(COUNT), COUNT)
#define			STR_FIT(STR)						array_fit(&STR, 1)
#define			ESTR_AT(TYPE, STR, IDX)				*(TYPE*)array_at(&(STR), IDX)

#define			STR_ALLOC(STR, LEN)				ESTR_ALLOC(char, STR, 0, LEN)
#define			STR_COPY(STR, DATA, LEN)		ESTR_ALLOC(char, STR, DATA, LEN)
#define			STR_AT(STR, IDX)				ESTR_AT(char, STR, IDX)

#define			WSTR_ALLOC(STR, LEN)			ESTR_ALLOC(wchar_t, STR, 0, LEN)
#define			WSTR_COPY(STR, DATA, LEN)		ESTR_ALLOC(wchar_t, STR, DATA, LEN)
#define			WSTR_AT(STR, IDX)				ESTR_AT(wchar_t, STR, IDX)
#endif


//double-linked LIST of identical size arrays		append-only, no mid-insertion
#if 1
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4200)//no default-constructor for struct with zero-length array
#endif
typedef struct DNodeStruct
{
	struct DNodeStruct *prev, *next;
	unsigned char data[];
} DNodeHeader, *DNodeHandle;
#ifdef _MSC_VER
#pragma warning(pop)
#endif
typedef struct DListStruct
{
	DNodeHandle i, f;
	size_t
		objsize,	//size of one contained object
		objpernode,	//object count per node,		recommended value 128
		nnodes,		//node count
		nobj;		//total object count
	void (*destructor)(void*);
} DList, *DListHandle;
void			dlist_init(DListHandle list, size_t objsize, size_t objpernode, void (*destructor)(void*));
void			dlist_copy(DListHandle dst, DListHandle src);
void			dlist_clear(DListHandle list);
void			dlist_appendtoarray(DListHandle list, ArrayHandle *dst);
void			dlist_appendtoarrayandclear(DListHandle list, ArrayHandle *dst);

void*			dlist_push_back1(DListHandle list, const void *obj);//shallow copy of obj
void*			dlist_push_back(DListHandle list, const void *data, size_t count);
void*			dlist_back(DListHandle list);//returns address of last object
void			dlist_pop_back(DListHandle list);

//iterator: seamlessly iterate through contained objects
typedef struct DListIteratorStruct
{
	DListHandle list;
	DNodeHandle node;
	size_t obj_idx;
} DListIterator, *DListItHandle;
void			dlist_first(DListHandle list, DListItHandle it);
void			dlist_last(DListHandle list, DListItHandle it);
void*			dlist_it_deref(DListItHandle it);
int				dlist_it_inc(DListItHandle it);
int				dlist_it_dec(DListItHandle it);
#endif

#ifdef __cplusplus
}
#endif
#endif//INC_AWM_ARRAY_H
