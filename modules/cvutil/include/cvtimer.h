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

#define CV_TIMER_START()                        \
  static CvTimer timer;                         \
  timer.start()

#define CV_TIMER_RESTART()                      \
  timer.restart()

#if 1
#  define CV_TIMER_SHOW()                                         \
  fprintf(stderr, "%s: ", __FUNCTION__);                          \
  fprintf(stderr, "elapsed: %fms\n", timer.elapsed()*1000.0f)
#else
#  define CV_TIMER_SHOW() {}
#endif

#endif // __CV_TIMER_H__
