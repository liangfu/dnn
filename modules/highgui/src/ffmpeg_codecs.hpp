/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(WIN32) || defined(__MINGW32__)
// some versions of FFMPEG assume a C99 compiler, and don't define INT64_C
#include <stdint.h>

// some versions of FFMPEG assume a C99 compiler, and don't define INT64_C
#ifndef INT64_C
#define INT64_C(c) (c##LL)
#endif

#ifndef UINT64_C
#define UINT64_C(c) (c##ULL)
#endif

#include <errno.h>
#endif

#ifdef WIN32
  #include <libavformat/avformat.h>
#else

// if the header path is not specified explicitly, let's deduce it
#if !defined HAVE_FFMPEG_AVCODEC_H && !defined HAVE_LIBAVCODEC_AVCODEC_H

#if defined(HAVE_GENTOO_FFMPEG)
  #define HAVE_LIBAVFORMAT_AVFORMAT_H 1
#elif defined HAVE_FFMPEG
  #define HAVE_FFMPEG_AVFORMAT_H 1
#endif

#if defined(HAVE_FFMPEG_AVFORMAT_H)
  #include <ffmpeg/avformat.h>
#endif

#if defined(HAVE_LIBAVFORMAT_AVFORMAT_H)
  #include <libavformat/avformat.h>
#endif

#endif

#endif

#ifdef __cplusplus
}
#endif

#ifndef MKTAG
#define MKTAG(a,b,c,d) (a | (b << 8) | (c << 16) | (d << 24))
#endif

#ifndef CALC_FFMPEG_VERSION
#define CALC_FFMPEG_VERSION(a,b,c) ( a<<16 | b<<8 | c )
#endif

// required to look up the correct codec ID depending on the FOURCC code,
// this is just a snipped from the file riff.c from ffmpeg/libavformat
typedef struct AVCodecTag {
    int id;
    unsigned int tag;
} AVCodecTag;

#if LIBAVCODEC_BUILD > CALC_FFMPEG_VERSION(55,1,0)
#define CV_CODEC_ID_H264            AV_CODEC_ID_H264 
#define CV_CODEC_ID_H263            AV_CODEC_ID_H263     
#define CV_CODEC_ID_H263P           AV_CODEC_ID_H263P    
#define CV_CODEC_ID_H263I           AV_CODEC_ID_H263I    
#define CV_CODEC_ID_H261            AV_CODEC_ID_H261     
#define CV_CODEC_ID_H263P           AV_CODEC_ID_H263P    
#define CV_CODEC_ID_MPEG4           AV_CODEC_ID_MPEG4    
#define CV_CODEC_ID_MSMPEG4V3       AV_CODEC_ID_MSMPEG4V3
#define CV_CODEC_ID_MSMPEG4V2       AV_CODEC_ID_MSMPEG4V2
#define CV_CODEC_ID_MSMPEG4V1       AV_CODEC_ID_MSMPEG4V1 
#define CV_CODEC_ID_WMV1            AV_CODEC_ID_WMV1            
#define CV_CODEC_ID_WMV2            AV_CODEC_ID_WMV2            
#define CV_CODEC_ID_DVVIDEO         AV_CODEC_ID_DVVIDEO         
#define CV_CODEC_ID_MPEG1VIDEO      AV_CODEC_ID_MPEG1VIDEO      
#define CV_CODEC_ID_MPEG2VIDEO      AV_CODEC_ID_MPEG2VIDEO      
#define CV_CODEC_ID_MPEG1VIDEO      AV_CODEC_ID_MPEG1VIDEO      
#define CV_CODEC_ID_MJPEG           AV_CODEC_ID_MJPEG           
#define CV_CODEC_ID_LJPEG           AV_CODEC_ID_LJPEG           
#define CV_CODEC_ID_HUFFYUV         AV_CODEC_ID_HUFFYUV         
#define CV_CODEC_ID_FFVHUFF         AV_CODEC_ID_FFVHUFF         
#define CV_CODEC_ID_CYUV            AV_CODEC_ID_CYUV            
#define CV_CODEC_ID_RAWVIDEO        AV_CODEC_ID_RAWVIDEO        
#define CV_CODEC_ID_INDEO3          AV_CODEC_ID_INDEO3          
#define CV_CODEC_ID_VP3             AV_CODEC_ID_VP3             
#define CV_CODEC_ID_ASV1            AV_CODEC_ID_ASV1            
#define CV_CODEC_ID_ASV2            AV_CODEC_ID_ASV2            
#define CV_CODEC_ID_VCR1            AV_CODEC_ID_VCR1            
#define CV_CODEC_ID_FFV1            AV_CODEC_ID_FFV1            
#define CV_CODEC_ID_XAN_WC4         AV_CODEC_ID_XAN_WC4         
#define CV_CODEC_ID_MSRLE           AV_CODEC_ID_MSRLE           
#define CV_CODEC_ID_MSVIDEO1        AV_CODEC_ID_MSVIDEO1        
#define CV_CODEC_ID_CINEPAK         AV_CODEC_ID_CINEPAK         
#define CV_CODEC_ID_TRUEMOTION1     AV_CODEC_ID_TRUEMOTION1     
#define CV_CODEC_ID_MSZH            AV_CODEC_ID_MSZH            
#define CV_CODEC_ID_ZLIB            AV_CODEC_ID_ZLIB            
#define CV_CODEC_ID_4XM             AV_CODEC_ID_4XM             
#define CV_CODEC_ID_FLV1            AV_CODEC_ID_FLV1            
#define CV_CODEC_ID_SVQ1            AV_CODEC_ID_SVQ1            
#define CV_CODEC_ID_TSCC            AV_CODEC_ID_TSCC            
#define CV_CODEC_ID_ULTI            AV_CODEC_ID_ULTI            
#define CV_CODEC_ID_VIXL            AV_CODEC_ID_VIXL            
#define CV_CODEC_ID_QPEG            AV_CODEC_ID_QPEG            
#define CV_CODEC_ID_QPEG            AV_CODEC_ID_QPEG            
#define CV_CODEC_ID_QPEG            AV_CODEC_ID_QPEG            
#define CV_CODEC_ID_WMV3            AV_CODEC_ID_WMV3            
#define CV_CODEC_ID_LOCO            AV_CODEC_ID_LOCO            
#define CV_CODEC_ID_THEORA          AV_CODEC_ID_THEORA          
#define CV_CODEC_ID_WNV1            AV_CODEC_ID_WNV1            
#define CV_CODEC_ID_AASC            AV_CODEC_ID_AASC            
#define CV_CODEC_ID_INDEO2          AV_CODEC_ID_INDEO2          
#define CV_CODEC_ID_FRAPS           AV_CODEC_ID_FRAPS           
#define CV_CODEC_ID_TRUEMOTION2     AV_CODEC_ID_TRUEMOTION2     
#define CV_CODEC_ID_FLASHSV         AV_CODEC_ID_FLASHSV         
#define CV_CODEC_ID_JPEGLS          AV_CODEC_ID_JPEGLS          
#define CV_CODEC_ID_VC1             AV_CODEC_ID_VC1             
#define CV_CODEC_ID_VC1             AV_CODEC_ID_VC1             
#define CV_CODEC_ID_CSCD            AV_CODEC_ID_CSCD            
#define CV_CODEC_ID_ZMBV            AV_CODEC_ID_ZMBV            
#define CV_CODEC_ID_KMVC            AV_CODEC_ID_KMVC            
#define CV_CODEC_ID_VP5             AV_CODEC_ID_VP5             
#define CV_CODEC_ID_VP6             AV_CODEC_ID_VP6             
#define CV_CODEC_ID_VP6F            AV_CODEC_ID_VP6F            
#define CV_CODEC_ID_JPEG2000        AV_CODEC_ID_JPEG2000        
#define CV_CODEC_ID_VMNC            AV_CODEC_ID_VMNC            
#define CV_CODEC_ID_TARGA           AV_CODEC_ID_TARGA           
#define CV_CODEC_ID_NONE            AV_CODEC_ID_NONE            
#else
#define CV_CODEC_ID_H264            CODEC_ID_H264 
#define CV_CODEC_ID_H263            CODEC_ID_H263     
#define CV_CODEC_ID_H263P           CODEC_ID_H263P    
#define CV_CODEC_ID_H263I           CODEC_ID_H263I    
#define CV_CODEC_ID_H261            CODEC_ID_H261     
#define CV_CODEC_ID_H263P           CODEC_ID_H263P    
#define CV_CODEC_ID_MPEG4           CODEC_ID_MPEG4    
#define CV_CODEC_ID_MSMPEG4V3       CODEC_ID_MSMPEG4V3
#define CV_CODEC_ID_MSMPEG4V2       CODEC_ID_MSMPEG4V2
#define CV_CODEC_ID_MSMPEG4V1       CODEC_ID_MSMPEG4V1 
#define CV_CODEC_ID_WMV1            CODEC_ID_WMV1            
#define CV_CODEC_ID_WMV2            CODEC_ID_WMV2            
#define CV_CODEC_ID_DVVIDEO         CODEC_ID_DVVIDEO         
#define CV_CODEC_ID_MPEG1VIDEO      CODEC_ID_MPEG1VIDEO      
#define CV_CODEC_ID_MPEG2VIDEO      CODEC_ID_MPEG2VIDEO      
#define CV_CODEC_ID_MPEG1VIDEO      CODEC_ID_MPEG1VIDEO      
#define CV_CODEC_ID_MJPEG           CODEC_ID_MJPEG           
#define CV_CODEC_ID_LJPEG           CODEC_ID_LJPEG           
#define CV_CODEC_ID_HUFFYUV         CODEC_ID_HUFFYUV         
#define CV_CODEC_ID_FFVHUFF         CODEC_ID_FFVHUFF         
#define CV_CODEC_ID_CYUV            CODEC_ID_CYUV            
#define CV_CODEC_ID_RAWVIDEO        CODEC_ID_RAWVIDEO        
#define CV_CODEC_ID_INDEO3          CODEC_ID_INDEO3          
#define CV_CODEC_ID_VP3             CODEC_ID_VP3             
#define CV_CODEC_ID_ASV1            CODEC_ID_ASV1            
#define CV_CODEC_ID_ASV2            CODEC_ID_ASV2            
#define CV_CODEC_ID_VCR1            CODEC_ID_VCR1            
#define CV_CODEC_ID_FFV1            CODEC_ID_FFV1            
#define CV_CODEC_ID_XAN_WC4         CODEC_ID_XAN_WC4         
#define CV_CODEC_ID_MSRLE           CODEC_ID_MSRLE           
#define CV_CODEC_ID_MSVIDEO1        CODEC_ID_MSVIDEO1        
#define CV_CODEC_ID_CINEPAK         CODEC_ID_CINEPAK         
#define CV_CODEC_ID_TRUEMOTION1     CODEC_ID_TRUEMOTION1     
#define CV_CODEC_ID_MSZH            CODEC_ID_MSZH            
#define CV_CODEC_ID_ZLIB            CODEC_ID_ZLIB            
#define CV_CODEC_ID_4XM             CODEC_ID_4XM             
#define CV_CODEC_ID_FLV1            CODEC_ID_FLV1            
#define CV_CODEC_ID_SVQ1            CODEC_ID_SVQ1            
#define CV_CODEC_ID_TSCC            CODEC_ID_TSCC            
#define CV_CODEC_ID_ULTI            CODEC_ID_ULTI            
#define CV_CODEC_ID_VIXL            CODEC_ID_VIXL            
#define CV_CODEC_ID_QPEG            CODEC_ID_QPEG            
#define CV_CODEC_ID_QPEG            CODEC_ID_QPEG            
#define CV_CODEC_ID_QPEG            CODEC_ID_QPEG            
#define CV_CODEC_ID_WMV3            CODEC_ID_WMV3            
#define CV_CODEC_ID_LOCO            CODEC_ID_LOCO            
#define CV_CODEC_ID_THEORA          CODEC_ID_THEORA          
#define CV_CODEC_ID_WNV1            CODEC_ID_WNV1            
#define CV_CODEC_ID_AASC            CODEC_ID_AASC            
#define CV_CODEC_ID_INDEO2          CODEC_ID_INDEO2          
#define CV_CODEC_ID_FRAPS           CODEC_ID_FRAPS           
#define CV_CODEC_ID_TRUEMOTION2     CODEC_ID_TRUEMOTION2     
#define CV_CODEC_ID_FLASHSV         CODEC_ID_FLASHSV         
#define CV_CODEC_ID_JPEGLS          CODEC_ID_JPEGLS          
#define CV_CODEC_ID_VC1             CODEC_ID_VC1             
#define CV_CODEC_ID_VC1             CODEC_ID_VC1             
#define CV_CODEC_ID_CSCD            CODEC_ID_CSCD            
#define CV_CODEC_ID_ZMBV            CODEC_ID_ZMBV            
#define CV_CODEC_ID_KMVC            CODEC_ID_KMVC            
#define CV_CODEC_ID_VP5             CODEC_ID_VP5             
#define CV_CODEC_ID_VP6             CODEC_ID_VP6             
#define CV_CODEC_ID_VP6F            CODEC_ID_VP6F            
#define CV_CODEC_ID_JPEG2000        CODEC_ID_JPEG2000        
#define CV_CODEC_ID_VMNC            CODEC_ID_VMNC            
#define CV_CODEC_ID_TARGA           CODEC_ID_TARGA           
#define CV_CODEC_ID_NONE            CODEC_ID_NONE            
#endif

const AVCodecTag codec_bmp_tags[] = {
    { CV_CODEC_ID_H264, MKTAG('H', '2', '6', '4') },
    { CV_CODEC_ID_H264, MKTAG('h', '2', '6', '4') },
    { CV_CODEC_ID_H264, MKTAG('X', '2', '6', '4') },
    { CV_CODEC_ID_H264, MKTAG('x', '2', '6', '4') },
    { CV_CODEC_ID_H264, MKTAG('a', 'v', 'c', '1') },
    { CV_CODEC_ID_H264, MKTAG('V', 'S', 'S', 'H') },

    { CV_CODEC_ID_H263, MKTAG('H', '2', '6', '3') },
    { CV_CODEC_ID_H263P, MKTAG('H', '2', '6', '3') },
    { CV_CODEC_ID_H263I, MKTAG('I', '2', '6', '3') }, /* intel h263 */
    { CV_CODEC_ID_H261, MKTAG('H', '2', '6', '1') },

    /* added based on MPlayer */
    { CV_CODEC_ID_H263P, MKTAG('U', '2', '6', '3') },
    { CV_CODEC_ID_H263P, MKTAG('v', 'i', 'v', '1') },

    { CV_CODEC_ID_MPEG4, MKTAG('F', 'M', 'P', '4') },
    { CV_CODEC_ID_MPEG4, MKTAG('D', 'I', 'V', 'X') },
    { CV_CODEC_ID_MPEG4, MKTAG('D', 'X', '5', '0') },
    { CV_CODEC_ID_MPEG4, MKTAG('X', 'V', 'I', 'D') },
    { CV_CODEC_ID_MPEG4, MKTAG('M', 'P', '4', 'S') },
    { CV_CODEC_ID_MPEG4, MKTAG('M', '4', 'S', '2') },
    { CV_CODEC_ID_MPEG4, MKTAG(0x04, 0, 0, 0) }, /* some broken avi use this */

    /* added based on MPlayer */
    { CV_CODEC_ID_MPEG4, MKTAG('D', 'I', 'V', '1') },
    { CV_CODEC_ID_MPEG4, MKTAG('B', 'L', 'Z', '0') },
    { CV_CODEC_ID_MPEG4, MKTAG('m', 'p', '4', 'v') },
    { CV_CODEC_ID_MPEG4, MKTAG('U', 'M', 'P', '4') },
    { CV_CODEC_ID_MPEG4, MKTAG('W', 'V', '1', 'F') },
    { CV_CODEC_ID_MPEG4, MKTAG('S', 'E', 'D', 'G') },

    { CV_CODEC_ID_MPEG4, MKTAG('R', 'M', 'P', '4') },

    { CV_CODEC_ID_MSMPEG4V3, MKTAG('D', 'I', 'V', '3') }, /* default signature when using MSMPEG4 */
    { CV_CODEC_ID_MSMPEG4V3, MKTAG('M', 'P', '4', '3') },

    /* added based on MPlayer */
    { CV_CODEC_ID_MSMPEG4V3, MKTAG('M', 'P', 'G', '3') },
    { CV_CODEC_ID_MSMPEG4V3, MKTAG('D', 'I', 'V', '5') },
    { CV_CODEC_ID_MSMPEG4V3, MKTAG('D', 'I', 'V', '6') },
    { CV_CODEC_ID_MSMPEG4V3, MKTAG('D', 'I', 'V', '4') },
    { CV_CODEC_ID_MSMPEG4V3, MKTAG('A', 'P', '4', '1') },
    { CV_CODEC_ID_MSMPEG4V3, MKTAG('C', 'O', 'L', '1') },
    { CV_CODEC_ID_MSMPEG4V3, MKTAG('C', 'O', 'L', '0') },

    { CV_CODEC_ID_MSMPEG4V2, MKTAG('M', 'P', '4', '2') },

    /* added based on MPlayer */
    { CV_CODEC_ID_MSMPEG4V2, MKTAG('D', 'I', 'V', '2') },

    { CV_CODEC_ID_MSMPEG4V1, MKTAG('M', 'P', 'G', '4') },

    { CV_CODEC_ID_WMV1, MKTAG('W', 'M', 'V', '1') },

    /* added based on MPlayer */
    { CV_CODEC_ID_WMV2, MKTAG('W', 'M', 'V', '2') },
    { CV_CODEC_ID_DVVIDEO, MKTAG('d', 'v', 's', 'd') },
    { CV_CODEC_ID_DVVIDEO, MKTAG('d', 'v', 'h', 'd') },
    { CV_CODEC_ID_DVVIDEO, MKTAG('d', 'v', 's', 'l') },
    { CV_CODEC_ID_DVVIDEO, MKTAG('d', 'v', '2', '5') },
    { CV_CODEC_ID_MPEG1VIDEO, MKTAG('m', 'p', 'g', '1') },
    { CV_CODEC_ID_MPEG1VIDEO, MKTAG('m', 'p', 'g', '2') },
    { CV_CODEC_ID_MPEG2VIDEO, MKTAG('m', 'p', 'g', '2') },
    { CV_CODEC_ID_MPEG2VIDEO, MKTAG('M', 'P', 'E', 'G') },
    { CV_CODEC_ID_MPEG1VIDEO, MKTAG('P', 'I', 'M', '1') },
    { CV_CODEC_ID_MPEG1VIDEO, MKTAG('V', 'C', 'R', '2') },
    { CV_CODEC_ID_MPEG1VIDEO, 0x10000001 },
    { CV_CODEC_ID_MPEG2VIDEO, 0x10000002 },
    { CV_CODEC_ID_MPEG2VIDEO, MKTAG('D', 'V', 'R', ' ') },
    { CV_CODEC_ID_MPEG2VIDEO, MKTAG('M', 'M', 'E', 'S') },
    { CV_CODEC_ID_MJPEG, MKTAG('M', 'J', 'P', 'G') },
    { CV_CODEC_ID_MJPEG, MKTAG('L', 'J', 'P', 'G') },
    { CV_CODEC_ID_LJPEG, MKTAG('L', 'J', 'P', 'G') },
    { CV_CODEC_ID_MJPEG, MKTAG('J', 'P', 'G', 'L') }, /* Pegasus lossless JPEG */
    { CV_CODEC_ID_MJPEG, MKTAG('M', 'J', 'L', 'S') }, /* JPEG-LS custom FOURCC for avi - decoder */
    { CV_CODEC_ID_MJPEG, MKTAG('j', 'p', 'e', 'g') },
    { CV_CODEC_ID_MJPEG, MKTAG('I', 'J', 'P', 'G') },
    { CV_CODEC_ID_MJPEG, MKTAG('A', 'V', 'R', 'n') },
    { CV_CODEC_ID_HUFFYUV, MKTAG('H', 'F', 'Y', 'U') },
    { CV_CODEC_ID_FFVHUFF, MKTAG('F', 'F', 'V', 'H') },
    { CV_CODEC_ID_CYUV, MKTAG('C', 'Y', 'U', 'V') },
    { CV_CODEC_ID_RAWVIDEO, 0 },
    { CV_CODEC_ID_RAWVIDEO, MKTAG('I', '4', '2', '0') },
    { CV_CODEC_ID_RAWVIDEO, MKTAG('Y', 'U', 'Y', '2') },
    { CV_CODEC_ID_RAWVIDEO, MKTAG('Y', '4', '2', '2') },
    { CV_CODEC_ID_RAWVIDEO, MKTAG('Y', 'V', '1', '2') },
    { CV_CODEC_ID_RAWVIDEO, MKTAG('U', 'Y', 'V', 'Y') },
    { CV_CODEC_ID_RAWVIDEO, MKTAG('I', 'Y', 'U', 'V') },
    { CV_CODEC_ID_RAWVIDEO, MKTAG('Y', '8', '0', '0') },
    { CV_CODEC_ID_RAWVIDEO, MKTAG('H', 'D', 'Y', 'C') },
    { CV_CODEC_ID_INDEO3, MKTAG('I', 'V', '3', '1') },
    { CV_CODEC_ID_INDEO3, MKTAG('I', 'V', '3', '2') },
    { CV_CODEC_ID_VP3, MKTAG('V', 'P', '3', '1') },
    { CV_CODEC_ID_VP3, MKTAG('V', 'P', '3', '0') },
    { CV_CODEC_ID_ASV1, MKTAG('A', 'S', 'V', '1') },
    { CV_CODEC_ID_ASV2, MKTAG('A', 'S', 'V', '2') },
    { CV_CODEC_ID_VCR1, MKTAG('V', 'C', 'R', '1') },
    { CV_CODEC_ID_FFV1, MKTAG('F', 'F', 'V', '1') },
    { CV_CODEC_ID_XAN_WC4, MKTAG('X', 'x', 'a', 'n') },
    { CV_CODEC_ID_MSRLE, MKTAG('m', 'r', 'l', 'e') },
    { CV_CODEC_ID_MSRLE, MKTAG(0x1, 0x0, 0x0, 0x0) },
    { CV_CODEC_ID_MSVIDEO1, MKTAG('M', 'S', 'V', 'C') },
    { CV_CODEC_ID_MSVIDEO1, MKTAG('m', 's', 'v', 'c') },
    { CV_CODEC_ID_MSVIDEO1, MKTAG('C', 'R', 'A', 'M') },
    { CV_CODEC_ID_MSVIDEO1, MKTAG('c', 'r', 'a', 'm') },
    { CV_CODEC_ID_MSVIDEO1, MKTAG('W', 'H', 'A', 'M') },
    { CV_CODEC_ID_MSVIDEO1, MKTAG('w', 'h', 'a', 'm') },
    { CV_CODEC_ID_CINEPAK, MKTAG('c', 'v', 'i', 'd') },
    { CV_CODEC_ID_TRUEMOTION1, MKTAG('D', 'U', 'C', 'K') },
    { CV_CODEC_ID_MSZH, MKTAG('M', 'S', 'Z', 'H') },
    { CV_CODEC_ID_ZLIB, MKTAG('Z', 'L', 'I', 'B') },
    // { CV_CODEC_ID_SNOW, MKTAG('S', 'N', 'O', 'W') },
    { CV_CODEC_ID_4XM, MKTAG('4', 'X', 'M', 'V') },
    { CV_CODEC_ID_FLV1, MKTAG('F', 'L', 'V', '1') },
    { CV_CODEC_ID_SVQ1, MKTAG('s', 'v', 'q', '1') },
    { CV_CODEC_ID_TSCC, MKTAG('t', 's', 'c', 'c') },
    { CV_CODEC_ID_ULTI, MKTAG('U', 'L', 'T', 'I') },
    { CV_CODEC_ID_VIXL, MKTAG('V', 'I', 'X', 'L') },
    { CV_CODEC_ID_QPEG, MKTAG('Q', 'P', 'E', 'G') },
    { CV_CODEC_ID_QPEG, MKTAG('Q', '1', '.', '0') },
    { CV_CODEC_ID_QPEG, MKTAG('Q', '1', '.', '1') },
    { CV_CODEC_ID_WMV3, MKTAG('W', 'M', 'V', '3') },
    { CV_CODEC_ID_LOCO, MKTAG('L', 'O', 'C', 'O') },
    { CV_CODEC_ID_THEORA, MKTAG('t', 'h', 'e', 'o') },
#if LIBAVCODEC_VERSION_INT>0x000409
    { CV_CODEC_ID_WNV1, MKTAG('W', 'N', 'V', '1') },
    { CV_CODEC_ID_AASC, MKTAG('A', 'A', 'S', 'C') },
    { CV_CODEC_ID_INDEO2, MKTAG('R', 'T', '2', '1') },
    { CV_CODEC_ID_FRAPS, MKTAG('F', 'P', 'S', '1') },
    { CV_CODEC_ID_TRUEMOTION2, MKTAG('T', 'M', '2', '0') },
#endif
#if LIBAVCODEC_VERSION_INT>((50<<16)+(1<<8)+0)
    { CV_CODEC_ID_FLASHSV, MKTAG('F', 'S', 'V', '1') },
    { CV_CODEC_ID_JPEGLS,MKTAG('M', 'J', 'L', 'S') }, /* JPEG-LS custom FOURCC for avi - encoder */
    { CV_CODEC_ID_VC1, MKTAG('W', 'V', 'C', '1') },
    { CV_CODEC_ID_VC1, MKTAG('W', 'M', 'V', 'A') },
    { CV_CODEC_ID_CSCD, MKTAG('C', 'S', 'C', 'D') },
    { CV_CODEC_ID_ZMBV, MKTAG('Z', 'M', 'B', 'V') },
    { CV_CODEC_ID_KMVC, MKTAG('K', 'M', 'V', 'C') },
#endif
#if LIBAVCODEC_VERSION_INT>((51<<16)+(11<<8)+0)
    { CV_CODEC_ID_VP5, MKTAG('V', 'P', '5', '0') },
    { CV_CODEC_ID_VP6, MKTAG('V', 'P', '6', '0') },
    { CV_CODEC_ID_VP6, MKTAG('V', 'P', '6', '1') },
    { CV_CODEC_ID_VP6, MKTAG('V', 'P', '6', '2') },
    { CV_CODEC_ID_VP6F, MKTAG('V', 'P', '6', 'F') },
    { CV_CODEC_ID_JPEG2000, MKTAG('M', 'J', '2', 'C') },
    { CV_CODEC_ID_VMNC, MKTAG('V', 'M', 'n', 'c') },
#endif
#if LIBAVCODEC_VERSION_INT>=((51<<16)+(49<<8)+0)
// this tag seems not to exist in older versions of FFMPEG
    { CV_CODEC_ID_TARGA, MKTAG('t', 'g', 'a', ' ') },
#endif
    { CV_CODEC_ID_NONE, 0 },
};
