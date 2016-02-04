// Selects sub-array (no data is copied)
CV_IMPL  CvMat*
cvGetSubRect( const CvArr* arr, CvMat* submat, CvRect rect )
{
    CvMat* res = 0;

    CV_FUNCNAME( "cvGetRect" );

    __BEGIN__;

    CvMat stub, *mat = (CvMat*)arr;

    if( !CV_IS_MAT( mat ))
        CV_CALL( mat = cvGetMat( mat, &stub ));

    if( !submat )
        CV_ERROR( CV_StsNullPtr, "" );

    if( (rect.x|rect.y|rect.width|rect.height) < 0 )
        CV_ERROR( CV_StsBadSize, "" );

    if( rect.x + rect.width > mat->cols ||
        rect.y + rect.height > mat->rows )
        CV_ERROR( CV_StsBadSize, "" );

    {
    submat->data.ptr = mat->data.ptr + (size_t)rect.y*mat->step +
                       rect.x*CV_ELEM_SIZE(mat->type);
    submat->step = mat->step & (rect.height > 1 ? -1 : 0);
    submat->type =
        (mat->type & (rect.width < mat->cols ? ~CV_MAT_CONT_FLAG : -1)) |
        (submat->step == 0 ? CV_MAT_CONT_FLAG : 0);
    submat->rows = rect.height;
    submat->cols = rect.width;
    submat->refcount = 0;
    res = submat;
    }

    __END__;

    return res;
}

