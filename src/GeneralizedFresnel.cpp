#include "GeneralizedFresnel.h" 

using namespace std;
using namespace boost::numeric::odeint;

GeneralizedFresnel::GeneralizedFresnel(double theta0, double k, double dk, 
	int order, double slowing_coef)
{
	init_theta = theta0;
	k0 = k;
	k1 = dk;
	n = order;
	alpha = slowing_coef;
}

void GeneralizedFresnel::generalized_fresnel_ode_fct(const state_type &state , 
	state_type &dstate , double t )
{
    dstate[0] = alpha*pow(t,n)*cos(init_theta+k0*t+k1*pow(t,2)/2);
    dstate[1] = alpha*pow(t,n)*sin(init_theta+k0*t+k1*pow(t,2)/2);
}

void GeneralizedFresnel::build_generalized_fresnel(const state_type &state ,const double t)
{
	if (t == 1.0)
	{
	    C_i = state[0];
	    S_i = state[1];
	}

}

void GeneralizedFresnel::generalized_fresnel_fct_i()
{
	using namespace std::placeholders;
	state_type init_state;
	init_state = {0,0};
	integrate_const(runge_kutta4< state_type >(), 
		std::bind(&GeneralizedFresnel::generalized_fresnel_ode_fct, this, _1, _2, _3),
		init_state,0.0,1.0,0.01, std::bind(&GeneralizedFresnel::build_generalized_fresnel, this, _1, _2));
}

void GeneralizedFresnel::generalized_fresnel_fct()
{
	for(int i = 0; i < n; i++)
	{
		GeneralizedFresnel F_i(init_theta,k0,k1,i,alpha);

		F_i.generalized_fresnel_fct_i();

		C.push_back(F_i.C_i);
		S.push_back(F_i.S_i);
	}
}

// int main(int argc, char **argv)
// {
// 	GeneralizedFresnel F(-0.785398,2.6799,-3.3598,3);

// 	F.generalized_fresnel_fct();

// 	for(int i = 0; i < F.C.size(); i++)
// 	{
// 		cout << F.C[i] << '\t' << F.S[i] << endl;
// 	}
	
// }


