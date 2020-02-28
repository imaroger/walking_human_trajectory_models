#ifndef GENERALIZEDFRESNEL_H 
#define GENERALIZEDFRESNEL_H 

#include <iostream>
#include <boost/array.hpp>
#include <boost/numeric/odeint.hpp> 

class GeneralizedFresnel
{

private:

	double k0;
	double k1;
	double init_theta;
	int n;	
	int i;
	double S_i;
	double C_i;
	typedef boost::array< double , 3 > state_type;
	double alpha;
	
public:
	
	std::vector<double> S;
	std::vector<double> C;	

	GeneralizedFresnel(double theta0, double k, double dk, int order, double slowing_coef);

	void generalized_fresnel_ode_fct(const state_type &state , state_type &dstate , double t );

	void build_generalized_fresnel(const state_type &state ,const double t);

	void generalized_fresnel_fct_i();

	void generalized_fresnel_fct();
	
};

#endif