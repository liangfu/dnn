/**
 * @file   cvunionfind.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Thu Jul  4 14:47:07 2013
 * 
 * @brief  
 * 
 * 
 */
#ifndef __CV_UNION_FIND_H__
#define __CV_UNION_FIND_H___

class CvUnionFind
{
  int* p;
  int* rank;
  int size;
  int* LowestVtxArray;

 public :
  CvUnionFind(int sz) 
  { 
    size = sz;
    p = new int[sz];
    rank = new int[sz];
    LowestVtxArray = new int[sz];
  }

  ~CvUnionFind() 
  {
    delete p;
    delete rank;
    delete LowestVtxArray;
  }

  void Clean()
  {
    int i;
    for (i = 0 ; i<size ; i++) {
      p[i]=0; rank[i]=0; LowestVtxArray[i]=0;
    }
  }

  void MakeSet(int x) 
  {
    p[x] = x ;
    rank[x] = 0;
  }
	
  void Union(int x , int y)
  {
    Link(FindSet(x) , FindSet(y));
  }
	
  void Link(int x , int y)
  {
    if (rank[x] > rank[y]) { p[y] = x; }
    else { p[x] = y; }
    if (rank[x] == rank[y]) { rank[y]++; }
  }

  int FindSet(int x)
  {
    if (x != p[x]) { p[x] = FindSet(p[x]); }
    return p[x];
  }

  void setLowestVertex(int v , int vid)
  {	
    LowestVtxArray[FindSet(v)] = vid; 
  }

  int getLowestVertex(int v)
  {
    return LowestVtxArray[FindSet(v)];
  }

  void setHighestVertex(int v , int vid)
  {
    LowestVtxArray[FindSet(v)] = vid; 
  }

  int getHighestVertex(int v)
  {
    return LowestVtxArray[FindSet(v)];
  }
};

#endif // __CV_UNION_FIND_H___
