#!/usr/bin/env python

import os,sys,re

def main():
  folders=['../include','../src']
  for folder in folders:
    realfolder=os.getcwd()+'/'+folder;
    filelist=filter(lambda x:x[-2:]=='.h' or x[-4:]=='.cpp',os.listdir(realfolder))
    filelist=map(lambda x:realfolder+'/'+x,filelist)
    for fname in filelist:
      fp=open(fname,'r')
      ctx=fp.read()
      res=re.sub(' --replace me-- ',' --replaced-- ',ctx)
      fp.close()
      fout=open(fname,'w')
      fout.write(res)
      fout.close

if __name__=="__main__":
  main()
