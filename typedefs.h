/*************************************************************************
 * typedefs.h - 
 *
 * $Id$
 *
 * LICENSE:
 * GPL v3.0 (see gpl-3.0.txt for details)
 *
 * AUTHOR:
 * Gregoire Malandain (gregoire.malandain@inria.fr)
 * 
 * CREATION DATE: 
 * June, 9 1998
 *
 *
 *
 */

#ifndef _typedefs_h_
#define _typedefs_h_

#ifdef __cplusplus
extern "C" {
#endif




/* Differents type coding for images and buffers.
 */
typedef enum {
  TYPE_UNKNOWN /* unknown type */,
  MYUCHAR, //UCHAR  /* unsigned char */,
  MYSCHAR, //SCHAR  /* signed char */,
  MYUSHORT,	//USHORT /* unsigned short int */,
  MYSSHORT, //SSHORT /* signed short int */,
  MYUINT, //UINT   /* unsigned int */,
  MYINT, //INT    /* signed int */,
  MYULINT, //ULINT  /* unsigned long int */,
  MYFLOAT, //FLOAT  /* float */,
  MYDOUBLE	//DOUBLE  /* double */
} ImageType, bufferType;

typedef char               s8;
typedef unsigned char      u8;
typedef short int          s16;
typedef unsigned short int u16;
typedef int                i32;
typedef int                s32;
typedef unsigned int       u32;
typedef unsigned long int  u64;
typedef float              r32;
typedef double             r64;





/* Typedef Booleen
 */
#ifdef False
   #undef False
#endif
#ifdef True
   #undef True
#endif
typedef enum {
  False = 0,
  True = 1
} typeBoolean;


#ifdef __cplusplus
}
#endif

#endif
