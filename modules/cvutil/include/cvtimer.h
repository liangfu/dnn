/**
 * @file   cvtimer.h
 * @author Liangfu Chen <chenclf@gmail.com>
 * @date   Fri Jun  7 22:42:17 2013
 * 
 * @brief  
 * 
 * 
 */

#ifndef __CV_TIMER_H__
#define __CV_TIMER_H__

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

class CvTimer
{
 public:
  void start()	// Constructor
  {
#ifdef WIN32
    QueryPerformanceFrequency(&m_CounterFrequency);
    QueryPerformanceCounter(&m_LastCount);
#else
    gettimeofday(&m_LastCount, 0);
#endif
  }
	
  // Resets timer (difference) to zero
  inline void restart() 
  {
#ifdef WIN32
    QueryPerformanceCounter(&m_LastCount);
#else
    gettimeofday(&m_LastCount, 0);
#endif
  }					
	
  // Get elapsed time in seconds
  float elapsed()
  {
    // Get the current count
#ifdef WIN32
    LARGE_INTEGER lCurrent;
    QueryPerformanceCounter(&lCurrent);

    return float((lCurrent.QuadPart - m_LastCount.QuadPart) /
                 double(m_CounterFrequency.QuadPart));
#else
    timeval lcurrent;
    gettimeofday(&lcurrent, 0);
    float fSeconds = (float)(lcurrent.tv_sec - m_LastCount.tv_sec);
    float fFraction =
        (float)(lcurrent.tv_usec - m_LastCount.tv_usec) * 0.000001f;
    return fSeconds + fFraction;
#endif
  }	
	
 protected:
#ifdef WIN32
  LARGE_INTEGER m_CounterFrequency;
  LARGE_INTEGER m_LastCount;
#else
  timeval m_LastCount;
#endif
};

// warning: this function is not thread safe
static char * time2str(float elapsed)
{
  static char res[80];
  int h = int(elapsed/3600.f)%24;
  int m = int(elapsed/60.f)%60;
  int s = int(elapsed)%60;
  float ms = float(elapsed-h*3600-m*60-s)*1000.f;
  if (h>0){sprintf(res,"%d hour %d min %d sec",h,m,s);}
  else if (m>0){sprintf(res,"%d min %d sec %.0f ms",m,s,ms);}
  else if (s>0){sprintf(res,"%d sec %.0f ms",s,ms);}
  else if (ms>0){sprintf(res,"%.2f ms",ms);}
  else{sprintf(res,"0 ms");}
  return res;
}

static char * time2str_concise(float elapsed)
{
  static char res[80];
  int h = int(elapsed/3600.f)%24;
  int m = int(elapsed/60.f)%60;
  int s = int(elapsed)%60;
  float ms = float(elapsed-h*3600-m*60-s)*1000.f;
  if (h>0){sprintf(res,"%dh%dm",h,m);}
  else if (m>0){sprintf(res,"%dm%ds",m,s);}
  else if (s>0){sprintf(res,"%ds%.0fms",s,ms);}
  else if (ms>0){sprintf(res,"%.2fms",ms);}
  else{sprintf(res,"0 ms");}
  return res;
}

#define CV_TIMER_START()                        \
  static CvTimer timer;                         \
  timer.start()

#define CV_TIMER_RESTART()                      \
  timer.restart()

#if 1
#  define CV_TIMER_SHOW()                                           \
  do {                                                              \
  fprintf(stderr, "%s: ", __FUNCTION__);                            \
  float seconds = timer.elapsed();                                  \
  fprintf(stderr,"elapsed: %s\n",time2str(seconds));                \
  }while(false)
#else
#  define CV_TIMER_SHOW() {}
#endif

#endif // __CV_TIMER_H__
