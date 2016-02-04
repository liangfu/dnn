/**
 * @file   cvext_io.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Thu Dec 20 11:41:51 2012
 * 
 * @brief  
 * 
 * 
 */

#include <ctype.h>
#include <sys/stat.h>
#include "cvext_c.h"

#ifndef ANDROID
CVAPI(void) cvSaveImageEx(const char * id, const CvArr * arr, CvRect roi)
{
  static int iter=0;
  int i;
  char imgfn[1024],annfn[1024]; 
  if ( (id[strlen(id)-1]!='/') && (id[strlen(id)-1]!='\\') )
  { sprintf(annfn,"%s/annotation.txt",id); }
  else
  { sprintf(annfn,"%sannotation.txt",id); }
  for (i=0;;i++){
    if ( (id[strlen(id)-1]!='/') && (id[strlen(id)-1]!='\\') )
    { sprintf(imgfn,"%s/img%05d.png",id,i); }
    else
    { sprintf(imgfn,"%simg%05d.png",id,i); }
    if (!icvFileExist(imgfn)){
      cvSaveImage(imgfn,arr);
      FILE * fp = fopen(annfn,"a+");
      if (iter==0){fprintf(fp,"--\n");}
      fprintf(fp,"%s %d %d %d %d\n",imgfn,roi.x,roi.y,roi.width,roi.height);
      fclose(fp);
      break;
    }
  }
  iter++;
}

CVAPI(void) cvShowImageEx(const char * id, const CvArr * arr,
                          const int cm)
{
  CvMat * src, src_stub;
  double minval, maxval, maxdiff; CvPoint minloc, maxloc;
  int type = cvGetElemType(arr);
  CvMat * disp, * src_scaled;
  int i, j;
  
  if (!CV_IS_MAT(arr))
    src = cvGetMat(arr, &src_stub);
  else{
    src = (CvMat*)arr;
  }

  src = cvCloneMat(src);
  if ( (src->rows<10) || (src->rows<10) )
  {
    CvMat * orig = cvCloneMat(src);
    int scale=60./MIN(orig->rows, orig->cols);
    cvReleaseMat(&src);
    src = cvCreateMat(orig->rows*scale, orig->cols*scale,
                      CV_MAT_TYPE(orig->type));
    int m,n;
    if (CV_MAT_TYPE(src->type)==CV_64F){
      for (m=0;m<orig->rows;m++) {
        for (n=0;n<orig->cols;n++) {
          for (i=0;i<scale;i++) {
            for (j=0;j<scale;j++) {
              CV_MAT_ELEM(*src, double, m*scale+i, n*scale+j) =
                  CV_MAT_ELEM(*orig, double, m, n);
            }
          }
        }
      }
    }
    cvReleaseMat(&orig);
  }

  switch (type){
    case CV_8U:
    case CV_8UC3:
      {
        cvShowImage(id, arr);
        break;
      }
    case CV_16S:
    case CV_32S:
    case CV_32F:
    case CV_64F:
      {
        src_scaled =
            cvCreateMat(src->rows, src->cols, CV_8U);
        disp = cvCreateMat(src->rows, src->cols, CV_8UC3);

        cvMinMaxLoc(src, &minval, &maxval, &minloc, &maxloc);
        maxdiff = maxval-minval;

        if (maxdiff==0.0) {
          cvSet(src_scaled, cvScalar(0));
        }else{
          if (cm==CV_CM_HSV)
          {
#if 0
          // hue range from 0 to 180 degree
          cvScale(src, src_scaled, 180/maxdiff, (-minval)*(180/maxdiff));
          cvThreshold(src_scaled, src_scaled, 179, 179, CV_THRESH_TRUNC);
          cvSubRS(src_scaled, cvScalar(179), src_scaled);
#else
          // hue range from 0 to 240 degree
          cvScale(src, src_scaled, 240/maxdiff, (-minval)*(240/maxdiff));
          cvThreshold(src_scaled, src_scaled, 239, 239, CV_THRESH_TRUNC);
          cvSubRS(src_scaled, cvScalar(239), src_scaled);
#endif
          }else if (cm==CV_CM_GRAY){
            cvScale(src, src_scaled, 256/maxdiff, (-minval)*(255/maxdiff));
          }
        }
        
        for (i = 0; i < src->rows; i++)
          for (j = 0; j < src->cols; j++)
          {
            CvScalar color;
            if (cm==CV_CM_HSV)
            {color = cvHue2BGR(CV_MAT_ELEM(*src_scaled, uchar, i, j));}
            else if (cm==CV_CM_GRAY)
            {
              color.val[0] = CV_MAT_ELEM(*src_scaled, uchar, i, j);
              color.val[1] = CV_MAT_ELEM(*src_scaled, uchar, i, j);
              color.val[2] = CV_MAT_ELEM(*src_scaled, uchar, i, j);
            }

            ((uchar*)(disp->data.ptr+(i*disp->step)))[j*3+0] = color.val[0];
            ((uchar*)(disp->data.ptr+(i*disp->step)))[j*3+1] = color.val[1];
            ((uchar*)(disp->data.ptr+(i*disp->step)))[j*3+2] = color.val[2];
          }

        cvShowImage(id, disp); 

        cvReleaseMat(&disp);
        cvReleaseMat(&src_scaled);
        break;
      }
    default:
      {
        fprintf(stderr, "Warning: unsupported type!");
        break;
      }
  }

  cvReleaseMat(&src);
}
#endif

CVAPI(void) cvPrintHeader(const char * fn, const CvArr * arr)
{
  FILE * fp = fopen(fn, "w");
  CvMat * mat, mat_stub;
  char c; int i, j;
  char strFormat[8],strHeaderDef[1024],strVar[1024];
  int type;
  CV_FUNCNAME("cvPrintHeader");
  __BEGIN__;
  if (!CV_IS_MAT(arr))
    mat = cvGetMat(arr, &mat_stub);
  else
    mat = (CvMat*)arr;

  type = CV_MAT_TYPE(mat->type);
  CV_ASSERT(fp!=NULL);
  CV_ASSERT(strncmp(fn+strlen(fn)-2,".h",2)==0);
  CV_ASSERT((type==CV_8U)||(type==CV_32S)||(type==CV_32F)||(type==CV_64F));
  
  // header definition
  strcpy(strHeaderDef, "__CV_XRC_");
  for (i = 0; i < strlen(fn); i++){
    c = fn[i];
    if (isalnum(c)){
      char cc[2]={toupper(c),'\0'};
      strcat(strHeaderDef, cc);
    } else {
      strcat(strHeaderDef, "_");
    }
  }
  strcat(strHeaderDef, "__");

  // variable name
  strcpy(strVar, "xrc");
  for (i = 0; i < strlen(fn); i++){
    c = fn[i];
    if (isalnum(c)){
      char cc[2]={toupper(c),'\0'};
      strcat(strVar, cc);
    }else if (c=='.'){
      break;
    }else{
      strcat(strVar, "_");
    }
  }

  fprintf(fp, "#ifndef %s\n", strHeaderDef);
  fprintf(fp, "#define %s\n", strHeaderDef);
  fprintf(fp, "\nstatic const %s %s[] = {\n",
          (type==CV_8U)?"uchar":(
              (type==CV_32S)?"int":(
                  (type==CV_32F)?"float":"double")),strVar);
  
  // data section
  switch (type){
    case CV_8U:
      for (i = 0; i < mat->rows-1; i++) { fprintf(fp, "  ");
        for (j = 0; j < mat->cols; j++){
          fprintf(fp, "%d, ", mat->data.ptr[i*mat->cols+j]);
        } fprintf(fp, "\n");
      }
      fprintf(fp, "  ");
      for (j = 0; j < mat->cols-1; j++){ 
        fprintf(fp, "%d, ", mat->data.ptr[(mat->rows-1)*mat->cols+j]);
      } fprintf(fp, "%d\n", mat->data.ptr[(mat->rows-1)*mat->cols+j]);
      break;
    case CV_32S:
      for (i = 0; i < mat->rows-1; i++) { fprintf(fp, "  ");
        for (j = 0; j < mat->cols; j++){
          fprintf(fp, "%d, ", mat->data.i[i*mat->cols+j]);
        } fprintf(fp, "\n");
      }
      fprintf(fp, "  ");
      for (j = 0; j < mat->cols-1; j++){ 
        fprintf(fp, "%d, ", mat->data.i[(mat->rows-1)*mat->cols+j]);
      } fprintf(fp, "%d\n", mat->data.i[(mat->rows-1)*mat->cols+j]);
      break;
    case CV_32F:
      for (i = 0; i < mat->rows-1; i++) { fprintf(fp, "  ");
        for (j = 0; j < mat->cols; j++){
          fprintf(fp, "%f, ", mat->data.fl[i*mat->cols+j]);
        } fprintf(fp, "\n");
      }
      fprintf(fp, "  ");
      for (j = 0; j < mat->cols-1; j++){
        fprintf(fp, "%f, ", mat->data.fl[(mat->rows-1)*mat->cols+j]);
      } fprintf(fp, "%f\n", mat->data.fl[(mat->rows-1)*mat->cols+j]);
      break;
    case CV_64F:
      for (i = 0; i < mat->rows-1; i++) { fprintf(fp, "  ");
        for (j = 0; j < mat->cols; j++){
          fprintf(fp, "%f, ", mat->data.db[i*mat->cols+j]);
        } fprintf(fp, "\n");
      }
      fprintf(fp, "  ");
      for (j = 0; j < mat->cols-1; j++){
        fprintf(fp, "%f, ", mat->data.db[(mat->rows-1)*mat->cols+j]);
      } fprintf(fp, "%f\n", mat->data.db[(mat->rows-1)*mat->cols+j]);
      break;
    default:
      fprintf(stderr, "Error: data type not supported\n");
      break;
  }

  fprintf(fp, "};\n\n#endif //%s\n", strHeaderDef);

  // release
  fclose(fp);
  __END__;
}

CVAPI(void) cvColorMapLUT(CvArr * _src, CvArr * _dst, CvArr * _lut)
{
  CvMat * src, src_stub, * dst, dst_stub, * lut, lut_stub;
  int i, j;
  
  CV_FUNCNAME("cvColorMapLUT");
  __BEGIN__;

  if (!CV_IS_MAT(_src)) {
    src = cvGetMat(_src, &src_stub);
  }else{
    src = (CvMat*)_src;
  }
  
  if (!CV_IS_MAT(_dst)) {
    dst = cvGetMat(_dst, &dst_stub);
  }else{
    dst = (CvMat*)_dst;
  }

  if (!CV_IS_MAT(_lut)) {
    lut = cvGetMat(_lut, &lut_stub);
  }else{
    lut = (CvMat*)_lut;
  }

  CV_ASSERT(CV_MAT_TYPE(src->type)==CV_8U);
  CV_ASSERT(CV_MAT_TYPE(dst->type)==CV_8UC3);
  
  typedef struct { uchar val[3]; } color_t;
  for (i = 0; i < src->rows; i++)
    for (j = 0; j < src->cols; j++)
    {
      color_t color;
      uchar intensity = CV_MAT_ELEM(*src, uchar, i, j);
      color.val[0] = CV_MAT_ELEM(*lut, uchar, intensity, 0);
      color.val[1] = CV_MAT_ELEM(*lut, uchar, intensity, 1);
      color.val[2] = CV_MAT_ELEM(*lut, uchar, intensity, 2);
      CV_MAT_ELEM(*dst, color_t, i, j) = color;
    }
  
  __END__;
}

CVAPI(CvMat *) icvLoadCSV(char * fn, int type)
{
  CvMat * mat = NULL;
  FILE * fp = fopen(fn,"r");
  char c,num[32],line[1<<12]; // 4096 char each line
  char tmp[2];
  int nr=0,nc=0; int i,err;char * tmpval;
  float fdata;
  // double arr[1024];
  assert(strcmp(fn+strlen(fn)-4,".csv")==0);

  if(fp==NULL)
  {
    perror("File open error");
    exit(EXIT_FAILURE);
  }

  nc=0;
  while(1)
  {
    tmpval=fgets(line, 1<<12, fp);
    if (feof(fp)) {break;}

    i=0; strcpy(num,"");
    if (nc==0)
    while (1)
    {
      c=line[i++];
      if (c!=','&&c!='\0'){
        sprintf(tmp, "%c\0", c);
        strcat(num,tmp);
      }else{
        if (c==','){
          nc++;
        }else{
          nc++;
          assert(c=='\0');
          break;
        }
        strcpy(num,"");
      }
    }

    nr++;
  }

  // fprintf(stderr, "dim: %dx%d\n", nr, nc);
  fclose(fp);
  fp = fopen(fn, "r");
  mat = cvCreateMat(nr, nc, type);

  nr = 0;
  while(1)
  {
    tmpval=fgets(line, 1<<12, fp);
    if (feof(fp)) {break;}

    i=0; nc=0; strcpy(num,"");

    while (1)
    {
      c=line[i++];
      if (c!=','&&c!='\0'){
        sprintf(tmp, "%c\0", c);
        strcat(num,tmp);
      }else{
        if (c==','){
          err=sscanf(num, "%f\n", &fdata); // assert(err!=EOF);
          if (type==CV_32F)
            CV_MAT_ELEM(*mat, float, nr, nc)=fdata;
          else if (type==CV_64F)
            CV_MAT_ELEM(*mat, double, nr, nc)=fdata;
          else if (type==CV_8U)
            CV_MAT_ELEM(*mat, uchar, nr, nc)=fdata;
          else if (type==CV_16S)
            CV_MAT_ELEM(*mat, short, nr, nc)=fdata;
          nc++;
        }else{
          err=sscanf(num, "%f\n", &fdata); // assert(err!=EOF);
          if (type==CV_32F)
            CV_MAT_ELEM(*mat, float, nr, nc)=fdata;
          else if (type==CV_64F)
            CV_MAT_ELEM(*mat, double, nr, nc)=fdata;
          else if (type==CV_8U)
            CV_MAT_ELEM(*mat, uchar, nr, nc)=fdata;
          else if (type==CV_16S)
            CV_MAT_ELEM(*mat, short, nr, nc)=fdata;
          nc++;
          assert(c=='\0');
          break;
        }
        strcpy(num,"");
      }
    }

    nr++;
  }

  // fprintf(stderr, "dim: %dx%d\n", nr, nc);

  fclose(fp);

  return mat;
}

CvMatND * icvLoadRAWIV(char * fn)
{
  int i;
  FILE * fp = fopen(fn, "r");
  if (!fp) {fprintf(stderr,"ERROR: fail to load %s\n",fn);return 0;}
  int type=-1; int bytes=-1; int dimsprod=-1;
  struct stat file_stat;

  float minlocs[3],maxlocs[3];
  unsigned int nverts,ncells;
  unsigned int dims[3];
  float origs[3],spans[3];
  fread(minlocs,4,3,fp);
  fread(maxlocs,4,3,fp);
  fread(&nverts,4,1,fp);
  fread(&ncells,4,1,fp);
  fread(dims,4,3,fp);
  fread(origs,4,3,fp);
  fread(spans,4,3,fp);
  // (6+2+9)*4=68 bits
  {
    uchar tmp; uchar * ptr = (uchar*)dims;
    for (i=0;i<12;i+=4){
      swap(&ptr[i],&ptr[i+3],1);swap(&ptr[i+1],&ptr[i+2],1);
    }
    swap(&dims[0],&dims[2],4);
  }
  dimsprod=dims[0]*dims[1]*dims[2];
  stat(fn,&file_stat);
  bytes=(file_stat.st_size-68)/dimsprod;
  type=(bytes==1)?CV_8U:((bytes==4)?CV_32F:CV_16S);
  CvMatND * data = cvCreateMatND(3,(int*)dims,type);
  fread(data->data.ptr,bytes,dimsprod,fp);
  if (bytes==2)
  {
    uchar * ptr = data->data.ptr;
    for (i=0;i<dimsprod*bytes;i+=2){
      swap(&ptr[i],&ptr[i+1],1);
    }
  }
  else if (bytes==4)
  {
    uchar * ptr = data->data.ptr;
    for (i=0;i<dimsprod*bytes;i+=4){
      swap(&ptr[i],&ptr[i+3],1);swap(&ptr[i+1],&ptr[i+2],1);
    }
  }
  fclose(fp);
  return data;
}

CVAPI(void) cvPrintf(FILE * fp, const char * format, CvArr * arr,
                     CvRect roi)
{
  CvMat src_stub, * src;
  int nr,nc,type,i=0,j=0,step;

  if (roi==cvRect(0,0,0,0)){
    src=cvCloneMat(cvGetMat(arr, &src_stub));
  }else{
    src=cvCloneMat(cvGetSubRect(arr, &src_stub, roi));
  }
  
  nr=src->rows;
  nc=src->cols;
  type=CV_MAT_TYPE(src->type);

  if (type==CV_32F)
  {
    float * fptr=src->data.fl;
    step=src->step/sizeof(float);
    for (i=0;i<nr;i++,fptr+=step)
    {
      for (j=0;j<nc;j++)
      {
        fprintf(fp, format, fptr[j]);
      }
      fprintf(fp, "\n");
    }
  }else if (type==CV_8U)
  {
    uchar * ptr=src->data.ptr;
    step=src->step/sizeof(uchar);
    for (i=0;i<nr;i++,ptr+=step)
    {
      for (j=0;j<nc;j++)
      {
        fprintf(fp, format, ptr[j]);
      }
      fprintf(fp, "\n");
    }
  }else if (type==CV_64F)
  {
    double * dptr=src->data.db;
    step=src->step/sizeof(double);
    for (i=0;i<nr;i++,dptr+=step)
    {
      for (j=0;j<nc;j++)
      {
        fprintf(fp, format, dptr[j]);
      }
      fprintf(fp, "\n");
    }
  }else if (type==CV_32S)
  {
    int * iptr=src->data.i;
    step=src->step/sizeof(int);
    for (i=0;i<nr;i++,iptr+=step)
    {
      for (j=0;j<nc;j++)
      {
        fprintf(fp, format, iptr[j]);
      }
      fprintf(fp, "\n");
    }
  }else{
    assert(false);
  }

  cvReleaseMat(&src);
}

/**
 * Which version of the C standard are we using?
 * Print some information to stdout.
 *
 * Mind the indentation!
 *
 * The following are required in the C11 Standard (mandatory macros).
 *   __STDC__			C89	C99	C11
 *   __STDC_HOSTED__			C99	C11
 *   __STDC_VERSION__		(C94)	C99	C11
 * The following are optional in the C11 Standard (environment macros).
 *   __STDC_ISO_10646__			C99	C11
 *   __STDC_MB_MIGHT_NEQ_WC__		C99	C11
 *   __STDC_UTF_16__				C11
 *   __STDC_UTF_32__				C11
 * The following are optional in the C11 Standard (conditional feature macros).
 *   __STDC_ANALYZABLE__			C11
 *   __STDC_IEC_559__			C99	C11
 *   __STDC_IEC_559_COMPLEX__		C99	C11
 *   __STDC_LIB_EXT1__				C11
 *   __STDC_NO_ATOMICS__			C11
 *   __STDC_NO_COMPLEX__			C11
 *   __STDC_NO_THREADS__			C11
 *   __STDC_NO_VLA__				C11
 *
 * The following are required in the C11 Standard (mandatory macros).
 *   __DATE__			C89	C99	C11
 *   __FILE__			C89	C99	C11
 *   __LINE__			C89	C99	C11
 *   __TIME__			C89	C99	C11
 */

void icvPrintCompilerInfo()
{
  #if defined (__cplusplus)
    printf("This is C++, version %d.\n", __cplusplus);
      /* The expected values would be
       *   199711L, for ISO/IEC 14882:1998 or 14882:2003
       */

  #elif defined(__STDC__)
    printf("This is standard C.\n");

    #if (__STDC__ == 1)
      printf("  The implementation is ISO-conforming.\n");
    #else
      printf("  The implementation is not ISO-conforming.\n");
      printf("    __STDC__ = %d\n", __STDC__);
    #endif

    #if defined(__STDC_VERSION__)
      #if (__STDC_VERSION__ >= 201112L)
        printf("This is C11.\n");
      #elif (__STDC_VERSION__ >= 199901L)
        printf("This is C99.\n");
      #elif (__STDC_VERSION__ >= 199409L)
        printf("This is C89 with amendment 1.\n");
      #else
        printf("This is C89 without amendment 1.\n");
        printf("  __STDC_VERSION__ = %ld\n", __STDC_VERSION__);
      #endif
    #else /* !defined(__STDC_VERSION__) */
      printf("This is C89.  __STDC_VERSION__ is not defined.\n");
    #endif

  #else   /* !defined(__STDC__) */
    printf("This is not standard C.  __STDC__ is not defined.\n");
  #endif

  #if defined(__STDC__) && defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
    #if defined(__STDC_HOSTED__)
      #if (__STDC_HOSTED__ == 1)
        printf("  This is a hosted implementation.\n");
      #elif (__STDC_HOSTED__ == 0)
        printf("  This is a freestanding implementation.\n");
      #else
        printf("  __STDC_HOSTED__ = %d is an unexpected value.\n",
               __STDC_HOSTED__);
      #endif
    #else
      printf("  __STDC_HOSTED__ is not defined.\n");
      printf("    This should indicate hosted or freestanding implementation.\n");
    #endif
  
    #if defined(__STDC_ISO_10646__)
      printf("  The wchar_t values conform to the ISO/IEC 10646 standard (Unicode) as of %ld.\n",
             __STDC_ISO_10646__);
    #else
      printf("  __STDC_ISO_10646__ is not defined.\n");
      printf("    The wchar_t values are implementation-defined.\n");
    #endif

    /* added in C99 Technical Corrigendum 3 */
    #if defined(__STDC_MB_MIGHT_NEQ_WC__)
      #if (__STDC_MB_MIGHT_NEQ_WC__ == 1)
        printf("  Wide and multibyte characters might not have the same codes.\n");
      #else
        printf("  __STDC_MB_MIGHT_NEQ_WC__ = %d is an unexpected value.\n",
               __STDC_MB_MIGHT_NEQ_WC__);
      #endif
    #else
      printf("  __STDC_MB_MIGHT_NEQ_WC__ is not defined.\n");
      printf("    Wide and multibyte characters should have the same codes.\n");
    #endif
  
    #if defined(__STDC_IEC_559__)
      #if (__STDC_IEC_559__ == 1)
        printf("  The floating-point implementation conforms to Annex F (IEC 60559 standard).\n");
      #else
        printf("  __STDC_IEC_559__ = %d is an unexpected value.\n",
               __STDC_IEC_559__);
      #endif
    #else
      printf("  __STDC_IEC_559__ is not defined.\n");
      printf("    The floating-point implementation does not conform to Annex F (IEC 60559 standard).\n");
    #endif

    #if defined(__STDC_IEC_559_COMPLEX__)
      #if (__STDC_IEC_559_COMPLEX__ == 1)
        printf("  The complex arithmetic implementation conforms to Annex G (IEC 60559 standard).\n");
      #else
        printf("  __STDC_IEC_559_COMPLEX__ = %d is an unexpected value.\n",
               __STDC_IEC_559_COMPLEX__);
      #endif
    #else
      printf("  __STDC_IEC_559_COMPLEX__ is not defined.\n");
      printf("    The complex arithmetic implementation does not conform to Annex G (IEC 60559 standard).\n");
    #endif
  #endif

  #if defined(__STDC__) && defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
    #if defined(__STDC_UTF_16__)
      #if (__STDC_UTF_16__ == 1)
	printf("  The char16_t values are UTF-16 encoded.\n");
      #else
	printf("  __STDC_UTF_16__ = %d is an unexpected value.\n",
	       __STDC_UTF_16__);
      #endif
    #else
      printf("  __STDC_UTF_16__ is not defined.\n");
      printf("    The char16_t values are implementation-defined.\n");
    #endif

    #if defined(__STDC_UTF_32__)
      #if (__STDC_UTF_32__ == 1)
	printf("  The char32_t values are UTF-32 encoded.\n");
      #else
	printf("  __STDC_UTF_32__ = %d is an unexpected value.\n",
	       __STDC_UTF_32__);
      #endif
    #else
      printf("  __STDC_UTF_32__ is not defined.\n");
      printf("    The char32_t values are implementation-defined.\n");
    #endif

    #if defined(__STDC_ANALYZABLE__)
      #if (__STDC_ANALYZABLE__ == 1)
	printf("  The compiler conforms to the specifications in Annex L (Analyzability).\n");
      #else
	printf("  __STDC_ANALYZABLE__ = %d is an unexpected value.\n",
	       __STDC_ANALYZABLE__);
      #endif
    #else
      printf("  __STDC_ANALYZABLE__ is not defined.\n");
      printf("    The compiler does not conform to the specifications in Annex L (Analyzability).\n");
    #endif

    #if defined(__STDC_LIB_EXT1__)
      printf("  The implementation supports the extensions defined in Annex K (Bounds-checking interfaces) as of %ld.\n",
             __STDC_LIB_EXT1__);
    #else
      printf("  __STDC_LIB_EXT1__ is not defined.\n");
      printf("    The implementation does not support the extensions defined in Annex K (Bounds-checking interfaces).\n");
    #endif

    #if defined(__STDC_NO_ATOMICS__)
      #if (__STDC_NO_ATOMICS__ == 1)
	printf("  The implementation does not support atomic types.\n");
      #else
	printf("  __STDC_NO_ATOMICS__ = %d is an unexpected value.\n",
	       __STDC_NO_ATOMICS__);
      #endif
    #else
      // printf("  __STDC_NO_ATOMICS__ is not defined.\n");
      printf("  The implementation supports atomic types and <stdatomic.h>.\n");
    #endif

    #if defined(__STDC_NO_COMPLEX__)
      #if (__STDC_NO_COMPLEX__ == 1)
	printf("  The implementation does not support complex types.\n");
      #else
	printf("  __STDC_NO_COMPLEX__ = %d is an unexpected value.\n",
	       __STDC_NO_COMPLEX__);
      #endif
      #if defined(__STDC_IEC_559_COMPLEX__)
	printf("  However, __STDC_IEC_559_COMPLEX__ is defined, and it should not be.\n");
      #endif
    #else
      // printf("  __STDC_NO_COMPLEX__ is not defined.\n");
      printf("  The implementation supports complex types and <complex.h>.\n");
    #endif

    #if defined(__STDC_NO_THREADS__)
      #if (__STDC_NO_THREADS__ == 1)
	printf("  The implementation does not support threads.\n");
      #else
	printf("  __STDC_NO_THREADS__ = %d is an unexpected value.\n",
	       __STDC_NO_THREADS__);
      #endif
    #else
      // printf("  __STDC_NO_THREADS__ is not defined.\n");
      printf("  The implementation supports threads and <threads.h>.\n");
    #endif

    #if defined(__STDC_NO_VLA__)
      #if (__STDC_NO_VLA__ == 1)
	printf("  The implementation does not support variable length arrays.\n");
      #else
	printf("  __STDC_NO_VLA__ = %d is an unexpected value.\n",
	       __STDC_NO_VLA__);
      #endif
    #else
      // printf("  __STDC_NO_VLA__ is not defined.\n");
      printf("  The implementation supports variable length arrays.\n");
    #endif

  #endif

  printf("\n");
}

