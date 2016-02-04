void cvPartialLeastSquares(Mat X,Mat Y,int FeatDim,Mat &T,Mat &P,Mat &U,Mat &Q,Mat &W, Mat &B)
/*********************************cvPartialLeastSquare****************************
*
* Function to compute the Partial Least Square decomposition of Data
*
* Input  -					1. X - Input Feature Matrix. X(i,:) stores the ith feature.
*							2. Y - Matrix containing class information such that Y = XW.
*								   If X(i,:) belongs to jth class, Y(j,:) should be 0 0 ... 1(at jth****    							   location)...
*								   X is a number of samples by feature dimension matrix.
*								   Y is a number of samples by no of classes -1 matrix.
*							3. FeatDim - Dimension of the output features.
*							4. W - Matrix whose columns contain the weight vectors.
* 
* Programmed by Jai Pillai. Please contact me at jsp@umiacs.umd.edu for comments, suggestions 
* and feed back.
* 
* This code is licensed under the Apache License, Version 2.0. Please cite the reference below
* in your publications if you use this code
* Jaishanker Pillai, "Partial Least Squares in C++", http://www.umiacs.umd.edu/~jsp.
*
**********************************************************************************/
{
	//Setting the termination criteria
	int nMaxIterations,nMaxOuter=1000;
	nMaxIterations = FeatDim;
	double TermCrit = 10e-15,TempVal;
	Mat tNorm;
	double MaxValX,MaxValY;
	int MaxIndexX,MaxIndexY;
	Mat TempX,TempY;

	//Matrices for storing the intermediate values.
	Mat tTemp,tNew,uTemp,wTemp,qTemp,pTemp,bTemp;

	//Allocating memory
	T.create(X.rows,nMaxIterations,CV_32F);
	P.create(X.cols,nMaxIterations,CV_32F);
	U.create(Y.rows,nMaxIterations,CV_32F);
	Q.create(Y.cols,nMaxIterations,CV_32F);
	W.create(X.cols,nMaxIterations,CV_32F);
	B.create(nMaxIterations,nMaxIterations,CV_32F);
	tTemp.create(X.rows,1,CV_32F);
	uTemp.create(Y.rows,1,CV_32F);

	for(int index1 = 0; index1 < nMaxIterations; index1++)
	{

		//Finding the column having the highest norm
		MaxValX = 0;MaxValY=0;MaxIndexX = -10;MaxIndexY=-10;
		TempX.create(X.rows,1,X.type());
		TempY.create(Y.rows,1,Y.type());
		for(int index3 = 0; index3 < X.cols; index3++)
		{
			for(int index2 = 0; index2 < X.rows; index2++)
			{
				TempX.at<float>(index2,0) = X.at<float>(index2,index3);
			}			
			if( norm(TempX) > MaxValX)
			{
				MaxValX = norm(TempX);
				MaxIndexX = index3;
			}
		}
		for(int index3 = 0; index3 < Y.cols; index3++)
		{
			for(int index2 = 0; index2 < Y.rows; index2++)
			{
				TempY.at<float>(index2,0) = Y.at<float>(index2,index3);
			}			
			if( norm(TempY) > MaxValY)
			{
				MaxValY = norm(TempY);
				MaxIndexY = index3;
			}
		}

		for(int index3 = 0; index3 < X.rows; index3++)
		{
			tTemp.at<float>(index3,0) = X.at<float>(index3,MaxIndexX);
			uTemp.at<float>(index3,0) = Y.at<float>(index3,MaxIndexY);
		}

		// Iteration for Outer Modelling
		for(int index2 = 0; index2 < nMaxOuter; index2++)
		{
			wTemp = X.t() * uTemp;
			wTemp = wTemp / norm(wTemp);
			tNew = X * wTemp;
			qTemp = Y.t() * tNew;
			qTemp = qTemp/ norm (qTemp);
			uTemp = Y * qTemp;

			TempVal = norm(tTemp - tNew);
			if(norm(tTemp - tNew) < TermCrit)
			{
			  break;
			}			      
			tTemp = tNew.clone();
		}    
    
		// Residual Deflation
		tNorm = tTemp.t()*tTemp;
		bTemp = uTemp.t()*tTemp/tNorm.at<float>(0,0);
		pTemp = X.t() * tTemp/tNorm.at<float>(0,0);
		X = X - tTemp * pTemp.t();
		Y = Y - bTemp.at<float>(0,0) * (tTemp * qTemp.t());


		// Saving Results to Outputs.
		for( int index3 = 0; index3 != X.rows; index3++)
		{
			T.at<float>(index3,index1) = tTemp.at<float>(index3,0);
			U.at<float>(index3,index1) = uTemp.at<float>(index3,0);
		}
		for( int index3 = 0; index3 != X.cols; index3++)
		{
			P.at<float>(index3,index1) = pTemp.at<float>(index3,0);
			W.at<float>(index3,index1) = wTemp.at<float>(index3,0);
		}

		for( int index3 = 0; index3 != qTemp.rows; index3++)
		{
			Q.at<float>(index3,index1) = qTemp.at<float>(index3,0);
		}
		B.at<float>(index1,index1) = bTemp.at<float>(0,0);

		// Checking the residue
		if ((norm(X) ==0)||(norm(Y) ==0))
		{
			break;
		}
		cout<<"Iteration Number is "<<index1<<endl;
	}
}
int main()
{
	Mat X,Y,T,P,U,Q,W,B;

	//Getting Input
	getFromMatlab(X,"X");
	getFromMatlab(Y,"Y");

	//Computing the Partial least Squares output.
	int OutputDimension = 100;
	cvPartialLeastSquares(X,Y,OutputDimension,T,P,U,Q,W,B);

	sendToMatlab(T,"TVC");
	sendToMatlab(P,"PVC");
	sendToMatlab(U,"UVC");
	sendToMatlab(Q,"QVC");
	sendToMatlab(W,"WVC");
	sendToMatlab(B,"BVC");


	return(0);
}
