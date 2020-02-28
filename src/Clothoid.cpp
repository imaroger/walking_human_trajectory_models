#include "Clothoid.h"
#include <chrono>
#include <ctime>

using namespace std;
using namespace boost::numeric::odeint;

Clothoid::Clothoid(double x0, double y0, double theta0, 
	double xf, double yf, double thetaf,double step, double slowing_coef)
{
	init_state = {x0,y0,theta0};
	final_state = {xf,yf,thetaf};
	dt = step;
	alpha = slowing_coef;
}

void Clothoid::build_clothoid_param()
{
	double dx = final_state[0]-init_state[0];
	double dy = final_state[1]-init_state[1];
	double R = sqrt(pow(dx,2)+pow(dy,2));
	double phi = atan2(dy,dx);

	double phi0 = normalizeAngle(init_state[2]-phi);
	double phif = normalizeAngle(final_state[2]-phi);
	double dphi = phif-phi0;

	double A_guess = guessA(phi0,phif);

	double A = findA(A_guess,dphi,phi0);

	GeneralizedFresnel F(phi0,dphi-A,2*A,1,alpha);

	F.generalized_fresnel_fct();

	L = R/F.C[0];

	if (L > 0)
	{
		k0 = (dphi-A)/L;
		k1 = 2*A/pow(L,2);
	}
	else
	{
		cout << "error: negative length" << endl;
	}
}

void Clothoid::clothoid_ode_fct(const state_type &state , state_type &dstate , double t )
{
    dstate[0] = alpha*cos(state[2]);
    dstate[1] = alpha*sin(state[2]);
    dstate[2] = k0+k1*t;
}

void Clothoid::build_clothoid( const state_type &state , const double t )
{
    x.push_back(state[0]);
    y.push_back(state[1]);
    theta.push_back(state[2]);    
}

void Clothoid::clothoid_fct()
{
	build_clothoid_param();

	cout << "k0 : " << k0 << " | k1 : " << k1 <<  " | L : " << L << endl;

	using namespace std::placeholders;
	integrate_const(runge_kutta4< state_type >(), std::bind(&Clothoid::clothoid_ode_fct, this, _1, _2, _3),
		init_state,0.0,L,dt, std::bind(&Clothoid::build_clothoid, this, _1, _2));
}

double Clothoid::normalizeAngle(double angle)
{
	double new_angle = angle;
	while (new_angle > M_PI)
	{
		new_angle -= 2*M_PI;
	}
	while (new_angle < -M_PI)
	{
		new_angle += 2*M_PI;
	}
	return new_angle;
}

double Clothoid::guessA(double phi0,double phif)
{
	double coef[] = {2.989696028701907,0.716228953608281,-0.458969738821509,
		-0.502821153340377,0.261062141752652,-0.045854475238709};
	double X = phi0/M_PI;
	double Y = phif/M_PI;	
	double xy = X*Y;	
	double A = (phi0+phif) * ( coef[0] + xy * ( coef[1] + xy * coef[2] )+
	(coef[3] + xy * coef[4]) * (pow(X,2)+pow(Y,2)) + coef[5] * (pow(X,4)+pow(Y,4)));
	return A;
}

double Clothoid::findA(double A_guess,double dphi,double phi0)
{
	double A = A_guess; 
	int n = 0;
	double k = 1;
	double dk;
	while (abs(k) > epsilon && n < 100)
	{
		GeneralizedFresnel F(phi0,dphi-A,2*A,3,alpha);
		F.generalized_fresnel_fct();
		vector<double> intS = F.S;
		vector<double> intC = F.C;

		k = intS[0];
		dk = intC[2]-intC[1];

		A  = A - k/dk;

		++n;
	}

	if (abs(k) > epsilon*10)
	{
		cout << "Newton iteration fails, k = " << k << endl;
		return 0;
	}
	else
	{
		return A;		
	}
}

int main(int argc, char **argv)
{
	clock_t t;
	t = clock();
	printf ("Calculating...\n");

	Clothoid C(1,-1,0.78,0.5,2,1.57,0.1,0.05);
	
	C.clothoid_fct();	  
	t = clock() - t;
	printf ("It took me %f seconds.\n",((float)t)/CLOCKS_PER_SEC);
}

