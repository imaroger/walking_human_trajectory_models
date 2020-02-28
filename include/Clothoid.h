#ifndef CLOTHOID_H 
#define CLOTHOID_H 

#include <iostream>
#include <boost/array.hpp>
#include <cmath>
#include <boost/numeric/odeint.hpp>
#include <stdlib.h>  
#include "GeneralizedFresnel.h"

class Clothoid
{

private:

	double dt;
	double epsilon = 1e-12;
	double alpha;

	typedef boost::array< double , 3 > state_type;
	state_type init_state;
	state_type final_state;

public:

	std::vector<double> x;
	std::vector<double> y;
	std::vector<double> theta;	
	double L;
	double k0;
	double k1;

	Clothoid(double x0, double y0, double theta0, 
		double xf, double yf, double thetaf, double step, double slowing_coef);

	void build_clothoid_param();

	void clothoid_ode_fct(const state_type &state , state_type &dstate , double t );

	void build_clothoid( const state_type &state , const double t );

	void clothoid_fct();

	double normalizeAngle(double angle);

	double guessA(double phi0,double phif);

	double findA(double A_guess,double dphi,double phi0);

};

#endif