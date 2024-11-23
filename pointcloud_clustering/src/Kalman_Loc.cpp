
//      Author: pdelapuente

#include "Kalman_Loc.h"

Kalman_Loc::Kalman_Loc(void)
: actualiza(1)
, borrar(1)
{
	noise_odom=0;
	
	pos_robot_kalman.SetSize(3,1);
	cov_robot_kalman.SetSize(3,3);
	Ri.SetSize(2,2);

//	map(0,0)=4; map(0,1)=2;
//	map(1,0)=6; map(1,1)=-2;
//	map(2,0)=7; map(2,1)=2;
//	map(3,0)=10; map(3,1)=6;

	sigma_odom=0.5f;   //valor inicial
	//sigma_odom=0.8f;
	//sigma_odom=2.5f;

	cov_robot_kalman.Null();
	pos_robot_kalman.Null();

	Ri=eye(2);
	//Ri*=pow(0.3f,2);
	Ri*=pow(0.3f,2);

	//Point2D borrar(3.0,20.0);
	//mapa.push_back(borrar);
}

Kalman_Loc::~Kalman_Loc(void)
{
}

Matrix Kalman_Loc::KalmanPos(float inc_odom_x, float inc_odom_y, float inc_odom_theta)//Next kalman position
{
	Matrix inc_odom(3,1);      
	inc_odom(0,0)=inc_odom_x;
	inc_odom(1,0)=inc_odom_y;
	inc_odom(2,0)=inc_odom_theta;

	//predicted state
	pos_robot_kalman=Comp(pos_robot_kalman,inc_odom);  
	Matrix Fx(3,3);
	Fx=J1(pos_robot_kalman,inc_odom);
	Matrix Fu(3,3);
	Fu=J2(pos_robot_kalman,inc_odom);

	Matrix cov_inc_odom(3,3);

	cov_inc_odom=eye(3);
	cov_inc_odom*=pow(sigma_odom,2);   

    cov_robot_kalman=Fx*cov_robot_kalman*(~Fx)+Fu*cov_inc_odom*(~Fu);

	return pos_robot_kalman;
}
	
void Kalman_Loc::KalmanUpdate(const std::vector<Point2D>& v)//se le pasan obs y corrige pos y cov Kalman
{	
//	Matrix obs(4,2);
	
/*FILE* f_time=fopen("TiemposUpdate.txt","a");
clock_t t;*/

	//FILE* fcov=fopen("Covarianzas","a");
	int tam_max=v.size()*2;
//fprintf(ftime,"Tamaño mapa %d Tam max %d\n",mapa.size(),tam_max);
	int tam=0;
	Matrix h(tam_max,1);
	Matrix Hx(tam_max,3);
	std::vector<Matrix> Hz;

	
	Matrix S;

	std::vector<int> obs_no_asociada;//indices de las obs no asociadas
//comienzo=clock();	
	//t=clock();
	
	Matrix hi(2,1);
	Matrix Hxi(2,3);Hxi(0,0)=Hxi(1,1)=1.0f;Hxi(1,0)=Hxi(0,1)=0.0f;
	Matrix Hzi(2,2);
	Matrix Si(2,2);

	Matrix hmin(2,1);
	Matrix Hxmin(2,3);
	Matrix Hzmin(2,2);
	Matrix Smin(2,2);

	if(borrar)
	{
		for(int k=0; k<mapa.size();k++)
		{
			if (PuntoEnPol(k))
			//if (PuntoNoVisto(k))
			{
				borrados.push_back(mapa[k]);
				mapa.erase(mapa.begin()+k);
				k--;

			}
						
		}
	}

	for(int i=0;i<v.size();i+=1) //las observaciones
	{
		float x_obs=v[i].xp;
		float y_obs=v[i].yp;
		if(x_obs*x_obs+y_obs*y_obs>36)
			continue;

		float dist_min=10000000000;
		
		float xr=pos_robot_kalman(0,0);
		float yr=pos_robot_kalman(1,0);
		float thr=pos_robot_kalman(2,0);
		float sen_alfa=sin(thr);
		float cos_alfa=cos(thr);
		float hx,hy;
		for (int j=0; j<mapa.size();j++)
		{
			hx=xr+x_obs*cos_alfa-y_obs*sen_alfa-mapa[j].xp;
			hy=yr+x_obs*sen_alfa+y_obs*cos_alfa-mapa[j].yp;
			if( hx*hx+hy*hy	> 0.1)   //12/9/07
			{
				/*Point2D pos(pos_robot_kalman(0,0),pos_robot_kalman(1,0));
				if ( hx*hx+hy*hy < 3 || squared_distance(mapa[j],pos)<3)
				{
					if(borrar)
					{
						if (PuntoEnPol(j))
						//if (PuntoNoVisto(j))
						{
							borrados.push_back(mapa[j]);
							mapa.erase(mapa.begin()+j);
							j--;

						}
					}
				}*/
			
				continue;
			}

			hi(0,0)=hx;
			hi(1,0)=hy;

			Hxi(0,2)=-x_obs*sen_alfa-y_obs*cos_alfa;
			Hxi(1,2)=x_obs*cos_alfa-y_obs*sen_alfa;

			Hzi(0,0)=cos_alfa; Hzi(0,1)=-sen_alfa;
			Hzi(1,0)=sen_alfa; Hzi(1,1)=cos_alfa; 
		
			Si=Hxi*cov_robot_kalman*(~Hxi)+	Hzi*Ri*(~Hzi);
			//Mahalanobis test
			Matrix Sinv=Si.Inv();
			Matrix M=(~hi)*Sinv*hi;
		//	fprintf(ftime,"Tiempo tras calcular S y M =%f\n",(clock()-comienzo)/(double)CLOCKS_PER_SEC);
			float dist=M(0,0);
			
			if(dist<dist_min && dist<0.3)			//if(dist<dist_min && dist<chi2inv(0.95,3))
			{										//NIS
			   dist_min=dist;
			   hmin=hi;
			   Hxmin=Hxi;
			   Hzmin=Hzi;
			   //Rmin=Ri;
			   //Smin=Si;
			}
			//if (PuntoNoVisto(j))
	/*		if (PuntoEnPol(j))
			{
				mapa.erase(mapa.begin()+j);
				j--;
			}*/
		
		}
	//	fprintf(ftime,"Tiempo tras asociar la observacion a punto del mapa=%f\n",(clock()-comienzo)/(double)CLOCKS_PER_SEC);

	
		if ( dist_min<10000000000)   //si ha superado el test de mahalanobis
		{	
			static int cont=0;
			if(cont==0)
			{
				cont++;
				h(tam,0)=hmin(0,0);
				h(tam+1,0)=hmin(1,0);
				Hx(tam,0)=Hxmin(0,0);	Hx(tam,1)=Hxmin(0,1);	Hx(tam,2)=Hxmin(0,2); 
				Hx(tam+1,0)=Hxmin(1,0); Hx(tam+1,1)=Hxmin(1,1); Hx(tam+1,2)=Hxmin(1,2); 

				tam+=2;	
				Hz.push_back(Hzmin);
			}
			else 
				cont=0;
		}
		else
		{
			obs_no_asociada.push_back(i);
		}
	//	fprintf(ftime,"Tiempo final para una observacion=%f\n",(clock()-comienzo)/(double)CLOCKS_PER_SEC);
	//	fprintf(ftime,"//////////////////////////////  \n");09
	}
//fprintf(ftime,"Tiempo final para las observaciones=%f\n",(clock()-comienzo)/(double)CLOCKS_PER_SEC);

	if(tam>0)//Kalman correction step
	{	
		Matrix PHt=cov_robot_kalman.Multiplica(~Hx,3,tam);
		S=Hx.Multiplica(PHt,tam,tam);   //2nd & 3rd =final dimensions

		for(int i=0;i<Hz.size();i++)
		{
			Matrix aux=Hz[i]*Ri*(~Hz[i]);
			S(i*2,i*2)+=aux(0,0); S(i*2,i*2+1)+=aux(0,1);
			S(i*2+1,i*2)+=aux(1,0); S(i*2+1,i*2+1)+=aux(1,1);
		}
		
		Matrix W=PHt*S.Inv();
		pos_robot_kalman=pos_robot_kalman-W.Multiplica(h,3,1);
		cov_robot_kalman=(eye(3)-W.Multiplica(Hx,3,3))*cov_robot_kalman;
		float pdiag=cov_robot_kalman(0,0);
		//fprintf(fcov,"%f\n",pdiag);
		//fclose(fcov);
	}
	//int tama=obs_no_asociada.size();
	if (actualiza)
		for(int k=0;k<obs_no_asociada.size();k++)
		{
			int ind=obs_no_asociada[k];
			Matrix obs(2,1);obs(0,0)=v[ind].xp;obs(1,0)=v[ind].yp;
			//Se utiliza la pos ya corregida
			Matrix pos_abs(2,1);
			pos_abs=Comp(pos_robot_kalman,obs);
			Point2D Point(pos_abs(0,0), pos_abs(1,0));
			mapa.push_back(Point);
		}
	
//fprintf(f_time,"Tiempo final=%f\n",(clock()-t)/(double)CLOCKS_PER_SEC);

//fclose(ftime);
	
}

/*Matrix Kalman_Loc::ObtainObservations(Matrix robot_pos, int rango)
{

	Matrix obs(0,2);
	for (int i=0;i<map.RowNo();i++)
	{
		Matrix med(2,1);
		med=Comp(InvTrans(robot_pos),~(map.SubMatrix(i,i+1,0,map.ColNo())));
		if(pow(med(0,0),2)+pow(med(1,0),2) < rango^2)
			obs.RowMatrixInsert(~(med),obs.RowNo());
	}

	return obs;
}*/

void Kalman_Loc::GuardarMapa(LPTSTR path)
{
	/*char* filepath = new char [1024];
	filepath=(char*) path;*/
	FILE* mapfile = _tfopen(path,TEXT("w"));
	for (int i=0; i<mapa.size();i++)
		fprintf(mapfile,"%f %f \n",mapa[i].xp,mapa[i].yp);
	fclose(mapfile);
	
}

void Kalman_Loc::LeerMapa(LPTSTR path)
{
	float x,y;
	FILE* mapfile = _tfopen(path,TEXT("r"));

	while (!feof(mapfile))
	{
		fscanf(mapfile,"%f %f \n",&x,&y);
		Point2D map_point(x,y);
		mapa.push_back(map_point);
	}
}


int Kalman_Loc::PointInPol(int k)
{
	//Solo es valido si el poligono es convexo
	for (int c=0;c<num_vert_pol-1;c++)
	{
		Point2D source=*(pol+c);
		Point2D target=*(pol+c+1);
		//Segment2D seg(*(pol+c),*(pol+c+1));
		//Vector2D vect_pol;
		//vect_pol.source=source;
		//vect_pol.target=target;
		Vector2D vect(source,mapa[k]);
		float ang1=atan2(target.yp-source.yp,target.xp-source.xp);
		float ang2=atan2(vect.target.yp-source.yp,vect.target.xp-source.xp);
		
		float erro_ang=ErrAng(ang1,ang2);
		erro_ang=AngRango(erro_ang);
		if (erro_ang <=0)     //si para alguno es negativo es que está fuera del polígono
			return 0;
	}

	Point2D source=*(pol+num_vert_pol-1);
	Point2D target=*(pol);
	Vector2D vect(source,mapa[k]);
	float ang1=atan2(target.yp-source.yp,target.xp-source.xp);
	float ang2=atan2(vect.target.yp-source.yp,vect.target.xp-source.xp);
	float erro_ang=ErrAng(ang1,ang2);
	erro_ang=AngRango(erro_ang);
	if (erro_ang <=0)     
		return 0;

	return 1;

}


int Kalman_Loc::PuntoNoVisto(int k)
{
	Point2D pos(pos_robot_kalman(0,0),pos_robot_kalman(1,0));
	if(squared_distance(pos,mapa[k])<min_dist*min_dist)
	{
		float ang1=atan2(mapa[k].yp-pos.yp,mapa[k].xp-pos.xp)+PI/2;
		float ang=ErrAng(0,ang1);
		ang=AngRango(ang);
		if(ang>0)
			return 1;
		else return 0;

	}
	else
		return 0;

}



int Kalman_Loc::PuntoEnPol(int k)
{
/*	Algoritmo radial
	float suma=0;
	for (int c=0;c<num_vert_pol;c++)
	{
		Point2D source=*(pol+c);
		Point2D target=*(pol+c+1);
		Segment2D seg(*(pol+c),*(pol+c+1));
		if (squared_distance(mapa[k],seg)<0.3*0.3)
			return 0;
		float ang1=atan2(seg.source.yp-mapa[k].yp,seg.source.xp-mapa[k].xp);   //Estos deberían ser angulos en sistema local mejor
		float ang2=atan2(seg.target.yp-mapa[k].yp,seg.target.xp-mapa[k].xp);
		float ang=ErrAng(ang1,ang2);
		suma+=ang;
	}
	if (suma<= PI)
		return 0;
	else
		return 1;*/

	for (int c=0;c<num_vert_pol;c++)
	{
		Point2D source=*(pol+c);
		Point2D target=*(pol+c+1);
		Segment2D seg(source,target);
		float dist2=squared_distance(mapa[k],seg);
		if (dist2<0.4*0.4)
			return 0;
	}

	Matrix mapalocal_k(3,1);
	mapalocal_k(0,0)=pos_robot_kalman(0,0)-mapa[k].xp;
	mapalocal_k(1,0)=pos_robot_kalman(1,0)-mapa[k].yp;
	mapalocal_k(2,0)=pos_robot_kalman(2,0);
	mapalocal_k=InvTrans(mapalocal_k);
	float xloc=mapalocal_k(0,0)-off_x;
	float yloc=mapalocal_k(1,0)-off_y;
	float ang=atan2(yloc,xloc)+ PI/2;
	
	for (int i=0;i<num_vert_pol-1;i++)
	{
			if (ang>=angulos[i] && ang<angulos[i+1])
			{
				Point2D source=*(pol+i);
				Point2D target=*(pol+i+1);
				Point2D pos(pos_robot_kalman(0,0),pos_robot_kalman(1,0));
				Point2D min=squared_distance(pos,source)<squared_distance(pos,target)?source:target;
				if (squared_distance(pos,mapa[k])<0.8*squared_distance(pos,min))
				{
					return 1;
				}
			}
	}
	
	return 0;

}
