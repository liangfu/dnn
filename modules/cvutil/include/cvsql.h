/**
 * @file   cvsql.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Tue Aug  6 11:58:37 2013
 * 
 * @brief  
 * 
 * 
 */
#ifndef __CV_SQL_H__
#define __CV_SQL_H__

#include <sqlite3.h>

static
int icvSqlCallback(void *notused, int argc, char **argv, char **col){
  int i;
  for(i=0; i<argc; i++){
    printf("%s = %s\n", col[i], argv[i] ? argv[i] : "NULL");
  }
  printf("\n");
  return 0;
}

static
int icvSqlCountCb(void * retval, int argc, char **argv, char **col){
  *(int*)retval=atoi(argv[0]);
  return 0;
}

class CvSql
{
  sqlite3 * db;

  int check_error(int rc){
    int failed=0;
    if ((rc!=SQLITE_OK)&&(rc!=SQLITE_DONE)){
      LOGE("%s",sqlite3_errmsg(db));
      failed=1;sqlite3_close(db);db=NULL;
    }else{failed=0;}
    return failed;
  }

 public:
  CvSql():db(NULL){}
  ~CvSql(){ close(); }

  int connected() { return (db!=NULL); }
  int connect(const char * dbname){
    int rc;
    rc = sqlite3_open(dbname, &db);
    
    check_error(rc);
#ifdef ANDROID
    LOGI("database %s connect success!",dbname);
#else
    if (rc!=SQLITE_OK){LOGE("can't connect database at location %s",dbname);}
    //else{LOGI("database connected at location %s",dbname);}
#endif //ANDROID
    return (rc==SQLITE_OK)?1:-1;
  }

  int close() {
    if (db) { int rc=sqlite3_close(db); check_error(rc); db=NULL;}
    return 1;
  }

  int execute(const char * stmt){
    int rc;char * errmsg;
    rc = sqlite3_exec(db,stmt,icvSqlCallback,0,&errmsg);
    check_error(rc);
    return rc;
  }

  int count(const char * table){
    if (!db){return -1;}
    int rc;char * errmsg;
    char stmt[0xff]; int retval=-1;
    sprintf(stmt,"select count(*) from %s", table);
    rc = sqlite3_exec(db,stmt,icvSqlCountCb,&retval,&errmsg);
    check_error(rc);
    return retval;
  }

  int table_exists(const char * table)
  {
    int rc;char * errmsg;
    char stmt[0xff]; int retval=-1;
    sprintf(stmt,"SELECT count(name) FROM sqlite_master "
            "WHERE type='table' AND name='%s'", table);
    rc = sqlite3_exec(db,stmt,icvSqlCountCb,&retval,&errmsg);
    check_error(rc);
    return retval;
  }

  int insert(const char * table, const char * name,
             CvMat * data, CvMat * img=0)
  {
    if (!db){return -1;}
    assert(CV_MAT_TYPE(data->type)==CV_32F);
    int failed=0;int rc;char * errmsg;
    static char sql[0xff];
    if (!img)
    {
      sprintf(sql,"insert into %s (name,data) values ('%s',?)",table,name);
    }else
    {
      sprintf(sql,"insert into %s (name,data,img) values ('%s',?,?)",
             table,name);
    }
    fprintf(stderr,"stmt:%s\n",sql);

    // convert floating-point array to binary string as sql statement
    assert(CV_MAT_TYPE(data->type)==CV_32F);
    assert(data->step/sizeof(float)==data->cols);

    sqlite3_stmt *stmt;
    int datsize = data->cols*data->rows*sizeof(float);
    fprintf(stderr,"datsize:%d\n",datsize);
    rc = sqlite3_prepare_v2(db,sql,-1,&stmt,NULL);
    failed=check_error(rc);

    if (!failed){
      rc = sqlite3_bind_blob(stmt,1,data->data.fl,datsize,SQLITE_TRANSIENT);
      failed=check_error(rc);
      if (img){
      assert(CV_MAT_TYPE(img->type)==CV_8U);
      int imgsize = img->cols*img->rows*sizeof(uchar);
      fprintf(stderr,"imgsize:%d\n",imgsize);
      rc = sqlite3_bind_blob(stmt,2,img->data.ptr,imgsize,SQLITE_TRANSIENT);
      failed=check_error(rc);
      }
    }

    if (!failed){ rc=sqlite3_step(stmt); failed=check_error(rc); }
    if (!failed){ rc=sqlite3_finalize(stmt); failed=check_error(rc); }
    
    return (rc==SQLITE_OK)?1:-1;
  }

  int select(const char * table){
    if (!db){return -1;}
    char stmt[0xff];
    int rc,nr,nc;char * errmsg;char ** result;
    sprintf(stmt,"select * from %s", table);
    rc = sqlite3_get_table(db,stmt,&result,&nr,&nc,&errmsg);
    //if (rc!=SQLITE_OK){fprintf(stderr,"ERROR: %s\n",errmsg);}
    check_error(rc);
    int i,j;
    for (i=0;i<(nr+1)*nc;i++){
      if (i%nc==nc-1) {
        if (i>nc){
          float aa[2];memcpy(aa,result[i],sizeof(aa));
          fprintf(stderr,"    %f,%f, ...",aa[0],aa[1]);
        }else{
          fprintf(stderr,"%18s ",result[i]);
        }
      }else{
        fprintf(stderr,"%8s ",result[i]);
      }
      if (i%nc==nc-1) {fprintf(stderr,"\n");};
    }
    fprintf(stderr,"INFO: table %s query success (%dx%d)!\n", table,nr,nc);
    sqlite3_free_table(result);
    return rc;
  }

  int query_int(const char * table, int rid, const char * col)
  {
    if (!db){return -1;}
    char sql[0xff];
    int rc,nr,nc;char * errmsg;
    sprintf(sql,"select %s from %s", col, table);
    char ** result;
    rc = sqlite3_get_table(db,sql,&result,&nr,&nc,&errmsg);
    check_error(rc);
    int retval=atoi(result[rid]);
    sqlite3_free_table(result);
    return retval;
  }
  
  int query(const char * table, const char * condition, const char * col,
            void * data, int bytes)
  {
    if (!db){return -1;}
    char sql[0xff];
    int rc,nr,nc;char * errmsg;
    sprintf(sql,"select %s from %s %s", col, table, condition);
    sqlite3_stmt * stmt;
    rc=sqlite3_prepare_v2(db,sql,strlen(sql)+1,&stmt,NULL); check_error(rc);
    sqlite3_step(stmt);
    memcpy(data,sqlite3_column_blob(stmt,0),bytes);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return 1;
  }
};

#endif //__CV_SQL_H__
