/* Created by Language version: 7.7.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__NMDA_Mg_T
#define _nrn_initial _nrn_initial__NMDA_Mg_T
#define nrn_cur _nrn_cur__NMDA_Mg_T
#define _nrn_current _nrn_current__NMDA_Mg_T
#define nrn_jacob _nrn_jacob__NMDA_Mg_T
#define nrn_state _nrn_state__NMDA_Mg_T
#define _net_receive _net_receive__NMDA_Mg_T 
#define kstates kstates__NMDA_Mg_T 
 
#define _threadargscomma_ /**/
#define _threadargsprotocomma_ /**/
#define _threadargs_ /**/
#define _threadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define Erev _p[0]
#define Erev_columnindex 0
#define gmax _p[1]
#define gmax_columnindex 1
#define mg _p[2]
#define mg_columnindex 2
#define i _p[3]
#define i_columnindex 3
#define g _p[4]
#define g_columnindex 4
#define rb _p[5]
#define rb_columnindex 5
#define rmb _p[6]
#define rmb_columnindex 6
#define rmu _p[7]
#define rmu_columnindex 7
#define rbMg _p[8]
#define rbMg_columnindex 8
#define rmc1b _p[9]
#define rmc1b_columnindex 9
#define rmc1u _p[10]
#define rmc1u_columnindex 10
#define rmc2b _p[11]
#define rmc2b_columnindex 11
#define rmc2u _p[12]
#define rmc2u_columnindex 12
#define U _p[13]
#define U_columnindex 13
#define Cl _p[14]
#define Cl_columnindex 14
#define D1 _p[15]
#define D1_columnindex 15
#define D2 _p[16]
#define D2_columnindex 16
#define O _p[17]
#define O_columnindex 17
#define UMg _p[18]
#define UMg_columnindex 18
#define ClMg _p[19]
#define ClMg_columnindex 19
#define D1Mg _p[20]
#define D1Mg_columnindex 20
#define D2Mg _p[21]
#define D2Mg_columnindex 21
#define OMg _p[22]
#define OMg_columnindex 22
#define DU _p[23]
#define DU_columnindex 23
#define DCl _p[24]
#define DCl_columnindex 24
#define DD1 _p[25]
#define DD1_columnindex 25
#define DD2 _p[26]
#define DD2_columnindex 26
#define DO _p[27]
#define DO_columnindex 27
#define DUMg _p[28]
#define DUMg_columnindex 28
#define DClMg _p[29]
#define DClMg_columnindex 29
#define DD1Mg _p[30]
#define DD1Mg_columnindex 30
#define DD2Mg _p[31]
#define DD2Mg_columnindex 31
#define DOMg _p[32]
#define DOMg_columnindex 32
#define _g _p[33]
#define _g_columnindex 33
#define _nd_area  *_ppvar[0]._pval
#define C	*_ppvar[2]._pval
#define _p_C	_ppvar[2]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  2;
 /* external NEURON variables */
 /* declaration of user functions */
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(Object* _ho) { void* create_point_process(int, Object*);
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt(void*);
 static double _hoc_loc_pnt(void* _vptr) {double loc_point_process(int, void*);
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(void* _vptr) {double has_loc_point(void*);
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(void* _vptr) {
 double get_loc_point_process(void*); return (get_loc_point_process(_vptr));
}
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 0,0
};
 static Member_func _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 0, 0
};
 /* declare global and static user variables */
#define Rmc2u Rmc2u_NMDA_Mg_T
 double Rmc2u = 0.06;
#define Rmc2b Rmc2b_NMDA_Mg_T
 double Rmc2b = 5e-08;
#define Rmc1u Rmc1u_NMDA_Mg_T
 double Rmc1u = 0.06;
#define Rmc1b Rmc1b_NMDA_Mg_T
 double Rmc1b = 5e-08;
#define Rmd2u Rmd2u_NMDA_Mg_T
 double Rmd2u = 0.06;
#define Rmd2b Rmd2b_NMDA_Mg_T
 double Rmd2b = 5e-08;
#define Rmd1u Rmd1u_NMDA_Mg_T
 double Rmd1u = 0.06;
#define Rmd1b Rmd1b_NMDA_Mg_T
 double Rmd1b = 5e-08;
#define RcMg RcMg_NMDA_Mg_T
 double RcMg = 0.0916;
#define RoMg RoMg_NMDA_Mg_T
 double RoMg = 0.0465;
#define Rr2Mg Rr2Mg_NMDA_Mg_T
 double Rr2Mg = 0.001932;
#define Rd2Mg Rd2Mg_NMDA_Mg_T
 double Rd2Mg = 0.002678;
#define Rr1Mg Rr1Mg_NMDA_Mg_T
 double Rr1Mg = 0.004002;
#define Rd1Mg Rd1Mg_NMDA_Mg_T
 double Rd1Mg = 0.02163;
#define RuMg RuMg_NMDA_Mg_T
 double RuMg = 0.06156;
#define RbMg RbMg_NMDA_Mg_T
 double RbMg = 0.01;
#define Rmu Rmu_NMDA_Mg_T
 double Rmu = 12.8;
#define Rmb Rmb_NMDA_Mg_T
 double Rmb = 5e-05;
#define Rc Rc_NMDA_Mg_T
 double Rc = 0.0916;
#define Ro Ro_NMDA_Mg_T
 double Ro = 0.0465;
#define Rr2 Rr2_NMDA_Mg_T
 double Rr2 = 0.0023;
#define Rd2 Rd2_NMDA_Mg_T
 double Rd2 = 0.004429;
#define Rr1 Rr1_NMDA_Mg_T
 double Rr1 = 0.00736;
#define Rd1 Rd1_NMDA_Mg_T
 double Rd1 = 0.02266;
#define Ru Ru_NMDA_Mg_T
 double Ru = 0.02016;
#define Rb Rb_NMDA_Mg_T
 double Rb = 0.01;
#define memb_fraction memb_fraction_NMDA_Mg_T
 double memb_fraction = 0.8;
#define rmd2u rmd2u_NMDA_Mg_T
 double rmd2u = 0;
#define rmd2b rmd2b_NMDA_Mg_T
 double rmd2b = 0;
#define rmd1u rmd1u_NMDA_Mg_T
 double rmd1u = 0;
#define rmd1b rmd1b_NMDA_Mg_T
 double rmd1b = 0;
#define valence valence_NMDA_Mg_T
 double valence = -2;
#define vmax vmax_NMDA_Mg_T
 double vmax = 100;
#define vmin vmin_NMDA_Mg_T
 double vmin = -120;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "vmin_NMDA_Mg_T", "mV",
 "vmax_NMDA_Mg_T", "mV",
 "Rb_NMDA_Mg_T", "/uM",
 "Ru_NMDA_Mg_T", "/ms",
 "Ro_NMDA_Mg_T", "/ms",
 "Rc_NMDA_Mg_T", "/ms",
 "Rd1_NMDA_Mg_T", "/ms",
 "Rr1_NMDA_Mg_T", "/ms",
 "Rd2_NMDA_Mg_T", "/ms",
 "Rr2_NMDA_Mg_T", "/ms",
 "Rmb_NMDA_Mg_T", "/uM",
 "Rmu_NMDA_Mg_T", "/ms",
 "Rmc1b_NMDA_Mg_T", "/uM",
 "Rmc1u_NMDA_Mg_T", "/ms",
 "Rmc2b_NMDA_Mg_T", "/uM",
 "Rmc2u_NMDA_Mg_T", "/ms",
 "Rmd1b_NMDA_Mg_T", "/uM",
 "Rmd1u_NMDA_Mg_T", "/ms",
 "Rmd2b_NMDA_Mg_T", "/uM",
 "Rmd2u_NMDA_Mg_T", "/ms",
 "RbMg_NMDA_Mg_T", "/uM",
 "RuMg_NMDA_Mg_T", "/ms",
 "RoMg_NMDA_Mg_T", "/ms",
 "RcMg_NMDA_Mg_T", "/ms",
 "Rd1Mg_NMDA_Mg_T", "/ms",
 "Rr1Mg_NMDA_Mg_T", "/ms",
 "Rd2Mg_NMDA_Mg_T", "/ms",
 "Rr2Mg_NMDA_Mg_T", "/ms",
 "rmd1b_NMDA_Mg_T", "/ms",
 "rmd1u_NMDA_Mg_T", "/ms",
 "rmd2b_NMDA_Mg_T", "/ms",
 "rmd2u_NMDA_Mg_T", "/ms",
 "Erev", "mV",
 "gmax", "pS",
 "mg", "mM",
 "i", "nA",
 "g", "pS",
 "rb", "/ms",
 "rmb", "/ms",
 "rmu", "/ms",
 "rbMg", "/ms",
 "rmc1b", "/ms",
 "rmc1u", "/ms",
 "rmc2b", "/ms",
 "rmc2u", "/ms",
 "C", "mM",
 0,0
};
 static double ClMg0 = 0;
 static double Cl0 = 0;
 static double D2Mg0 = 0;
 static double D1Mg0 = 0;
 static double D20 = 0;
 static double D10 = 0;
 static double OMg0 = 0;
 static double O0 = 0;
 static double UMg0 = 0;
 static double U0 = 0;
 static double delta_t = 1;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "vmin_NMDA_Mg_T", &vmin_NMDA_Mg_T,
 "vmax_NMDA_Mg_T", &vmax_NMDA_Mg_T,
 "valence_NMDA_Mg_T", &valence_NMDA_Mg_T,
 "memb_fraction_NMDA_Mg_T", &memb_fraction_NMDA_Mg_T,
 "Rb_NMDA_Mg_T", &Rb_NMDA_Mg_T,
 "Ru_NMDA_Mg_T", &Ru_NMDA_Mg_T,
 "Ro_NMDA_Mg_T", &Ro_NMDA_Mg_T,
 "Rc_NMDA_Mg_T", &Rc_NMDA_Mg_T,
 "Rd1_NMDA_Mg_T", &Rd1_NMDA_Mg_T,
 "Rr1_NMDA_Mg_T", &Rr1_NMDA_Mg_T,
 "Rd2_NMDA_Mg_T", &Rd2_NMDA_Mg_T,
 "Rr2_NMDA_Mg_T", &Rr2_NMDA_Mg_T,
 "Rmb_NMDA_Mg_T", &Rmb_NMDA_Mg_T,
 "Rmu_NMDA_Mg_T", &Rmu_NMDA_Mg_T,
 "Rmc1b_NMDA_Mg_T", &Rmc1b_NMDA_Mg_T,
 "Rmc1u_NMDA_Mg_T", &Rmc1u_NMDA_Mg_T,
 "Rmc2b_NMDA_Mg_T", &Rmc2b_NMDA_Mg_T,
 "Rmc2u_NMDA_Mg_T", &Rmc2u_NMDA_Mg_T,
 "Rmd1b_NMDA_Mg_T", &Rmd1b_NMDA_Mg_T,
 "Rmd1u_NMDA_Mg_T", &Rmd1u_NMDA_Mg_T,
 "Rmd2b_NMDA_Mg_T", &Rmd2b_NMDA_Mg_T,
 "Rmd2u_NMDA_Mg_T", &Rmd2u_NMDA_Mg_T,
 "RbMg_NMDA_Mg_T", &RbMg_NMDA_Mg_T,
 "RuMg_NMDA_Mg_T", &RuMg_NMDA_Mg_T,
 "RoMg_NMDA_Mg_T", &RoMg_NMDA_Mg_T,
 "RcMg_NMDA_Mg_T", &RcMg_NMDA_Mg_T,
 "Rd1Mg_NMDA_Mg_T", &Rd1Mg_NMDA_Mg_T,
 "Rr1Mg_NMDA_Mg_T", &Rr1Mg_NMDA_Mg_T,
 "Rd2Mg_NMDA_Mg_T", &Rd2Mg_NMDA_Mg_T,
 "Rr2Mg_NMDA_Mg_T", &Rr2Mg_NMDA_Mg_T,
 "rmd1b_NMDA_Mg_T", &rmd1b_NMDA_Mg_T,
 "rmd1u_NMDA_Mg_T", &rmd1u_NMDA_Mg_T,
 "rmd2b_NMDA_Mg_T", &rmd2b_NMDA_Mg_T,
 "rmd2u_NMDA_Mg_T", &rmd2u_NMDA_Mg_T,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 static void _hoc_destroy_pnt(void* _vptr) {
   destroy_point_process(_vptr);
}
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[3]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"NMDA_Mg_T",
 "Erev",
 "gmax",
 "mg",
 0,
 "i",
 "g",
 "rb",
 "rmb",
 "rmu",
 "rbMg",
 "rmc1b",
 "rmc1u",
 "rmc2b",
 "rmc2u",
 0,
 "U",
 "Cl",
 "D1",
 "D2",
 "O",
 "UMg",
 "ClMg",
 "D1Mg",
 "D2Mg",
 "OMg",
 0,
 "C",
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 34, _prop);
 	/*initialize range parameters*/
 	Erev = 5;
 	gmax = 500;
 	mg = 1;
  }
 	_prop->param = _p;
 	_prop->param_size = 34;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _NMDA_Mg_T_reg() {
	int _vectorized = 0;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 0,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 34, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "pointer");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 NMDA_Mg_T /Users/wybo/Code/NEAT_public/neat/tools/simtools/neuron/tmp/multichannel_test/mech/NMDA_Mg_T.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "kinetic NMDA receptor model";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 extern double *_getelm();
 
#define _MATELM1(_row,_col)	*(_getelm(_row + 1, _col + 1))
 
#define _RHS1(_arg) _coef1[_arg + 1]
 static double *_coef1;
 
#define _linmat1  1
 static void* _sparseobj1;
 static void* _cvsparseobj1;
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[10], _dlist1[10]; static double *_temp1;
 static int kstates();
 
static int kstates ()
 {_reset=0;
 {
   double b_flux, f_flux, _term; int _i;
 {int _i; double _dt1 = 1.0/dt;
for(_i=1;_i<10;_i++){
  	_RHS1(_i) = -_dt1*(_p[_slist1[_i]] - _p[_dlist1[_i]]);
	_MATELM1(_i, _i) = _dt1;
      
} }
 rb = Rb * ( 1e3 ) * C ;
   rbMg = RbMg * ( 1e3 ) * C ;
   rmb = Rmb * mg * ( 1e3 ) * exp ( ( v - 40.0 ) * valence * memb_fraction / 25.0 ) ;
   rmu = Rmu * exp ( ( - 1.0 ) * ( v - 40.0 ) * valence * ( 1.0 - memb_fraction ) / 25.0 ) ;
   rmc1b = Rmc1b * mg * ( 1e3 ) * exp ( ( v - 40.0 ) * valence * memb_fraction / 25.0 ) ;
   rmc1u = Rmc1u * exp ( ( - 1.0 ) * ( v - 40.0 ) * valence * ( 1.0 - memb_fraction ) / 25.0 ) ;
   rmc2b = Rmc2b * mg * ( 1e3 ) * exp ( ( v - 40.0 ) * valence * memb_fraction / 25.0 ) ;
   rmc2u = Rmc2u * exp ( ( - 1.0 ) * ( v - 40.0 ) * valence * ( 1.0 - memb_fraction ) / 25.0 ) ;
   rmd1b = Rmd1b * mg * ( 1e3 ) * exp ( ( v - 40.0 ) * valence * memb_fraction / 25.0 ) ;
   rmd1u = Rmd1u * exp ( ( - 1.0 ) * ( v - 40.0 ) * valence * ( 1.0 - memb_fraction ) / 25.0 ) ;
   rmd2b = Rmd2b * mg * ( 1e3 ) * exp ( ( v - 40.0 ) * valence * memb_fraction / 25.0 ) ;
   rmd2u = Rmd2u * exp ( ( - 1.0 ) * ( v - 40.0 ) * valence * ( 1.0 - memb_fraction ) / 25.0 ) ;
   /* ~ U <-> Cl ( rb , Ru )*/
 f_flux =  rb * U ;
 b_flux =  Ru * Cl ;
 _RHS1( 9) -= (f_flux - b_flux);
 _RHS1( 2) += (f_flux - b_flux);
 
 _term =  rb ;
 _MATELM1( 9 ,9)  += _term;
 _MATELM1( 2 ,9)  -= _term;
 _term =  Ru ;
 _MATELM1( 9 ,2)  -= _term;
 _MATELM1( 2 ,2)  += _term;
 /*REACTION*/
  /* ~ Cl <-> O ( Ro , Rc )*/
 f_flux =  Ro * Cl ;
 b_flux =  Rc * O ;
 _RHS1( 2) -= (f_flux - b_flux);
 _RHS1( 7) += (f_flux - b_flux);
 
 _term =  Ro ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 7 ,2)  -= _term;
 _term =  Rc ;
 _MATELM1( 2 ,7)  -= _term;
 _MATELM1( 7 ,7)  += _term;
 /*REACTION*/
  /* ~ Cl <-> D1 ( Rd1 , Rr1 )*/
 f_flux =  Rd1 * Cl ;
 b_flux =  Rr1 * D1 ;
 _RHS1( 2) -= (f_flux - b_flux);
 _RHS1( 6) += (f_flux - b_flux);
 
 _term =  Rd1 ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 6 ,2)  -= _term;
 _term =  Rr1 ;
 _MATELM1( 2 ,6)  -= _term;
 _MATELM1( 6 ,6)  += _term;
 /*REACTION*/
  /* ~ D1 <-> D2 ( Rd2 , Rr2 )*/
 f_flux =  Rd2 * D1 ;
 b_flux =  Rr2 * D2 ;
 _RHS1( 6) -= (f_flux - b_flux);
 _RHS1( 5) += (f_flux - b_flux);
 
 _term =  Rd2 ;
 _MATELM1( 6 ,6)  += _term;
 _MATELM1( 5 ,6)  -= _term;
 _term =  Rr2 ;
 _MATELM1( 6 ,5)  -= _term;
 _MATELM1( 5 ,5)  += _term;
 /*REACTION*/
  /* ~ O <-> OMg ( rmb , rmu )*/
 f_flux =  rmb * O ;
 b_flux =  rmu * OMg ;
 _RHS1( 7) -= (f_flux - b_flux);
 
 _term =  rmb ;
 _MATELM1( 7 ,7)  += _term;
 _term =  rmu ;
 _MATELM1( 7 ,0)  -= _term;
 /*REACTION*/
  /* ~ UMg <-> ClMg ( rbMg , RuMg )*/
 f_flux =  rbMg * UMg ;
 b_flux =  RuMg * ClMg ;
 _RHS1( 8) -= (f_flux - b_flux);
 _RHS1( 1) += (f_flux - b_flux);
 
 _term =  rbMg ;
 _MATELM1( 8 ,8)  += _term;
 _MATELM1( 1 ,8)  -= _term;
 _term =  RuMg ;
 _MATELM1( 8 ,1)  -= _term;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
  /* ~ ClMg <-> OMg ( RoMg , RcMg )*/
 f_flux =  RoMg * ClMg ;
 b_flux =  RcMg * OMg ;
 _RHS1( 1) -= (f_flux - b_flux);
 
 _term =  RoMg ;
 _MATELM1( 1 ,1)  += _term;
 _term =  RcMg ;
 _MATELM1( 1 ,0)  -= _term;
 /*REACTION*/
  /* ~ ClMg <-> D1Mg ( Rd1Mg , Rr1Mg )*/
 f_flux =  Rd1Mg * ClMg ;
 b_flux =  Rr1Mg * D1Mg ;
 _RHS1( 1) -= (f_flux - b_flux);
 _RHS1( 4) += (f_flux - b_flux);
 
 _term =  Rd1Mg ;
 _MATELM1( 1 ,1)  += _term;
 _MATELM1( 4 ,1)  -= _term;
 _term =  Rr1Mg ;
 _MATELM1( 1 ,4)  -= _term;
 _MATELM1( 4 ,4)  += _term;
 /*REACTION*/
  /* ~ D1Mg <-> D2Mg ( Rd2Mg , Rr2Mg )*/
 f_flux =  Rd2Mg * D1Mg ;
 b_flux =  Rr2Mg * D2Mg ;
 _RHS1( 4) -= (f_flux - b_flux);
 _RHS1( 3) += (f_flux - b_flux);
 
 _term =  Rd2Mg ;
 _MATELM1( 4 ,4)  += _term;
 _MATELM1( 3 ,4)  -= _term;
 _term =  Rr2Mg ;
 _MATELM1( 4 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
  /* ~ U <-> UMg ( rmc1b , rmc1u )*/
 f_flux =  rmc1b * U ;
 b_flux =  rmc1u * UMg ;
 _RHS1( 9) -= (f_flux - b_flux);
 _RHS1( 8) += (f_flux - b_flux);
 
 _term =  rmc1b ;
 _MATELM1( 9 ,9)  += _term;
 _MATELM1( 8 ,9)  -= _term;
 _term =  rmc1u ;
 _MATELM1( 9 ,8)  -= _term;
 _MATELM1( 8 ,8)  += _term;
 /*REACTION*/
  /* ~ Cl <-> ClMg ( rmc2b , rmc2u )*/
 f_flux =  rmc2b * Cl ;
 b_flux =  rmc2u * ClMg ;
 _RHS1( 2) -= (f_flux - b_flux);
 _RHS1( 1) += (f_flux - b_flux);
 
 _term =  rmc2b ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 1 ,2)  -= _term;
 _term =  rmc2u ;
 _MATELM1( 2 ,1)  -= _term;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
  /* ~ D1 <-> D1Mg ( rmd1b , rmd1u )*/
 f_flux =  rmd1b * D1 ;
 b_flux =  rmd1u * D1Mg ;
 _RHS1( 6) -= (f_flux - b_flux);
 _RHS1( 4) += (f_flux - b_flux);
 
 _term =  rmd1b ;
 _MATELM1( 6 ,6)  += _term;
 _MATELM1( 4 ,6)  -= _term;
 _term =  rmd1u ;
 _MATELM1( 6 ,4)  -= _term;
 _MATELM1( 4 ,4)  += _term;
 /*REACTION*/
  /* ~ D2 <-> D2Mg ( rmd2b , rmd2u )*/
 f_flux =  rmd2b * D2 ;
 b_flux =  rmd2u * D2Mg ;
 _RHS1( 5) -= (f_flux - b_flux);
 _RHS1( 3) += (f_flux - b_flux);
 
 _term =  rmd2b ;
 _MATELM1( 5 ,5)  += _term;
 _MATELM1( 3 ,5)  -= _term;
 _term =  rmd2u ;
 _MATELM1( 5 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
   /* U + Cl + D1 + D2 + O + UMg + ClMg + D1Mg + D2Mg + OMg = 1.0 */
 _RHS1(0) =  1.0;
 _MATELM1(0, 0) = 1;
 _RHS1(0) -= OMg ;
 _MATELM1(0, 3) = 1;
 _RHS1(0) -= D2Mg ;
 _MATELM1(0, 4) = 1;
 _RHS1(0) -= D1Mg ;
 _MATELM1(0, 1) = 1;
 _RHS1(0) -= ClMg ;
 _MATELM1(0, 8) = 1;
 _RHS1(0) -= UMg ;
 _MATELM1(0, 7) = 1;
 _RHS1(0) -= O ;
 _MATELM1(0, 5) = 1;
 _RHS1(0) -= D2 ;
 _MATELM1(0, 6) = 1;
 _RHS1(0) -= D1 ;
 _MATELM1(0, 2) = 1;
 _RHS1(0) -= Cl ;
 _MATELM1(0, 9) = 1;
 _RHS1(0) -= U ;
 /*CONSERVATION*/
   } return _reset;
 }
 
/*CVODE ode begin*/
 static int _ode_spec1() {_reset=0;{
 double b_flux, f_flux, _term; int _i;
 {int _i; for(_i=0;_i<10;_i++) _p[_dlist1[_i]] = 0.0;}
 rb = Rb * ( 1e3 ) * C ;
 rbMg = RbMg * ( 1e3 ) * C ;
 rmb = Rmb * mg * ( 1e3 ) * exp ( ( v - 40.0 ) * valence * memb_fraction / 25.0 ) ;
 rmu = Rmu * exp ( ( - 1.0 ) * ( v - 40.0 ) * valence * ( 1.0 - memb_fraction ) / 25.0 ) ;
 rmc1b = Rmc1b * mg * ( 1e3 ) * exp ( ( v - 40.0 ) * valence * memb_fraction / 25.0 ) ;
 rmc1u = Rmc1u * exp ( ( - 1.0 ) * ( v - 40.0 ) * valence * ( 1.0 - memb_fraction ) / 25.0 ) ;
 rmc2b = Rmc2b * mg * ( 1e3 ) * exp ( ( v - 40.0 ) * valence * memb_fraction / 25.0 ) ;
 rmc2u = Rmc2u * exp ( ( - 1.0 ) * ( v - 40.0 ) * valence * ( 1.0 - memb_fraction ) / 25.0 ) ;
 rmd1b = Rmd1b * mg * ( 1e3 ) * exp ( ( v - 40.0 ) * valence * memb_fraction / 25.0 ) ;
 rmd1u = Rmd1u * exp ( ( - 1.0 ) * ( v - 40.0 ) * valence * ( 1.0 - memb_fraction ) / 25.0 ) ;
 rmd2b = Rmd2b * mg * ( 1e3 ) * exp ( ( v - 40.0 ) * valence * memb_fraction / 25.0 ) ;
 rmd2u = Rmd2u * exp ( ( - 1.0 ) * ( v - 40.0 ) * valence * ( 1.0 - memb_fraction ) / 25.0 ) ;
 /* ~ U <-> Cl ( rb , Ru )*/
 f_flux =  rb * U ;
 b_flux =  Ru * Cl ;
 DU -= (f_flux - b_flux);
 DCl += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ Cl <-> O ( Ro , Rc )*/
 f_flux =  Ro * Cl ;
 b_flux =  Rc * O ;
 DCl -= (f_flux - b_flux);
 DO += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ Cl <-> D1 ( Rd1 , Rr1 )*/
 f_flux =  Rd1 * Cl ;
 b_flux =  Rr1 * D1 ;
 DCl -= (f_flux - b_flux);
 DD1 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ D1 <-> D2 ( Rd2 , Rr2 )*/
 f_flux =  Rd2 * D1 ;
 b_flux =  Rr2 * D2 ;
 DD1 -= (f_flux - b_flux);
 DD2 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ O <-> OMg ( rmb , rmu )*/
 f_flux =  rmb * O ;
 b_flux =  rmu * OMg ;
 DO -= (f_flux - b_flux);
 DOMg += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ UMg <-> ClMg ( rbMg , RuMg )*/
 f_flux =  rbMg * UMg ;
 b_flux =  RuMg * ClMg ;
 DUMg -= (f_flux - b_flux);
 DClMg += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ ClMg <-> OMg ( RoMg , RcMg )*/
 f_flux =  RoMg * ClMg ;
 b_flux =  RcMg * OMg ;
 DClMg -= (f_flux - b_flux);
 DOMg += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ ClMg <-> D1Mg ( Rd1Mg , Rr1Mg )*/
 f_flux =  Rd1Mg * ClMg ;
 b_flux =  Rr1Mg * D1Mg ;
 DClMg -= (f_flux - b_flux);
 DD1Mg += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ D1Mg <-> D2Mg ( Rd2Mg , Rr2Mg )*/
 f_flux =  Rd2Mg * D1Mg ;
 b_flux =  Rr2Mg * D2Mg ;
 DD1Mg -= (f_flux - b_flux);
 DD2Mg += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ U <-> UMg ( rmc1b , rmc1u )*/
 f_flux =  rmc1b * U ;
 b_flux =  rmc1u * UMg ;
 DU -= (f_flux - b_flux);
 DUMg += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ Cl <-> ClMg ( rmc2b , rmc2u )*/
 f_flux =  rmc2b * Cl ;
 b_flux =  rmc2u * ClMg ;
 DCl -= (f_flux - b_flux);
 DClMg += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ D1 <-> D1Mg ( rmd1b , rmd1u )*/
 f_flux =  rmd1b * D1 ;
 b_flux =  rmd1u * D1Mg ;
 DD1 -= (f_flux - b_flux);
 DD1Mg += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ D2 <-> D2Mg ( rmd2b , rmd2u )*/
 f_flux =  rmd2b * D2 ;
 b_flux =  rmd2u * D2Mg ;
 DD2 -= (f_flux - b_flux);
 DD2Mg += (f_flux - b_flux);
 
 /*REACTION*/
   /* U + Cl + D1 + D2 + O + UMg + ClMg + D1Mg + D2Mg + OMg = 1.0 */
 /*CONSERVATION*/
   } return _reset;
 }
 
/*CVODE matsol*/
 static int _ode_matsol1() {_reset=0;{
 double b_flux, f_flux, _term; int _i;
   b_flux = f_flux = 0.;
 {int _i; double _dt1 = 1.0/dt;
for(_i=0;_i<10;_i++){
  	_RHS1(_i) = _dt1*(_p[_dlist1[_i]]);
	_MATELM1(_i, _i) = _dt1;
      
} }
 rb = Rb * ( 1e3 ) * C ;
 rbMg = RbMg * ( 1e3 ) * C ;
 rmb = Rmb * mg * ( 1e3 ) * exp ( ( v - 40.0 ) * valence * memb_fraction / 25.0 ) ;
 rmu = Rmu * exp ( ( - 1.0 ) * ( v - 40.0 ) * valence * ( 1.0 - memb_fraction ) / 25.0 ) ;
 rmc1b = Rmc1b * mg * ( 1e3 ) * exp ( ( v - 40.0 ) * valence * memb_fraction / 25.0 ) ;
 rmc1u = Rmc1u * exp ( ( - 1.0 ) * ( v - 40.0 ) * valence * ( 1.0 - memb_fraction ) / 25.0 ) ;
 rmc2b = Rmc2b * mg * ( 1e3 ) * exp ( ( v - 40.0 ) * valence * memb_fraction / 25.0 ) ;
 rmc2u = Rmc2u * exp ( ( - 1.0 ) * ( v - 40.0 ) * valence * ( 1.0 - memb_fraction ) / 25.0 ) ;
 rmd1b = Rmd1b * mg * ( 1e3 ) * exp ( ( v - 40.0 ) * valence * memb_fraction / 25.0 ) ;
 rmd1u = Rmd1u * exp ( ( - 1.0 ) * ( v - 40.0 ) * valence * ( 1.0 - memb_fraction ) / 25.0 ) ;
 rmd2b = Rmd2b * mg * ( 1e3 ) * exp ( ( v - 40.0 ) * valence * memb_fraction / 25.0 ) ;
 rmd2u = Rmd2u * exp ( ( - 1.0 ) * ( v - 40.0 ) * valence * ( 1.0 - memb_fraction ) / 25.0 ) ;
 /* ~ U <-> Cl ( rb , Ru )*/
 _term =  rb ;
 _MATELM1( 9 ,9)  += _term;
 _MATELM1( 2 ,9)  -= _term;
 _term =  Ru ;
 _MATELM1( 9 ,2)  -= _term;
 _MATELM1( 2 ,2)  += _term;
 /*REACTION*/
  /* ~ Cl <-> O ( Ro , Rc )*/
 _term =  Ro ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 7 ,2)  -= _term;
 _term =  Rc ;
 _MATELM1( 2 ,7)  -= _term;
 _MATELM1( 7 ,7)  += _term;
 /*REACTION*/
  /* ~ Cl <-> D1 ( Rd1 , Rr1 )*/
 _term =  Rd1 ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 6 ,2)  -= _term;
 _term =  Rr1 ;
 _MATELM1( 2 ,6)  -= _term;
 _MATELM1( 6 ,6)  += _term;
 /*REACTION*/
  /* ~ D1 <-> D2 ( Rd2 , Rr2 )*/
 _term =  Rd2 ;
 _MATELM1( 6 ,6)  += _term;
 _MATELM1( 5 ,6)  -= _term;
 _term =  Rr2 ;
 _MATELM1( 6 ,5)  -= _term;
 _MATELM1( 5 ,5)  += _term;
 /*REACTION*/
  /* ~ O <-> OMg ( rmb , rmu )*/
 _term =  rmb ;
 _MATELM1( 7 ,7)  += _term;
 _MATELM1( 0 ,7)  -= _term;
 _term =  rmu ;
 _MATELM1( 7 ,0)  -= _term;
 _MATELM1( 0 ,0)  += _term;
 /*REACTION*/
  /* ~ UMg <-> ClMg ( rbMg , RuMg )*/
 _term =  rbMg ;
 _MATELM1( 8 ,8)  += _term;
 _MATELM1( 1 ,8)  -= _term;
 _term =  RuMg ;
 _MATELM1( 8 ,1)  -= _term;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
  /* ~ ClMg <-> OMg ( RoMg , RcMg )*/
 _term =  RoMg ;
 _MATELM1( 1 ,1)  += _term;
 _MATELM1( 0 ,1)  -= _term;
 _term =  RcMg ;
 _MATELM1( 1 ,0)  -= _term;
 _MATELM1( 0 ,0)  += _term;
 /*REACTION*/
  /* ~ ClMg <-> D1Mg ( Rd1Mg , Rr1Mg )*/
 _term =  Rd1Mg ;
 _MATELM1( 1 ,1)  += _term;
 _MATELM1( 4 ,1)  -= _term;
 _term =  Rr1Mg ;
 _MATELM1( 1 ,4)  -= _term;
 _MATELM1( 4 ,4)  += _term;
 /*REACTION*/
  /* ~ D1Mg <-> D2Mg ( Rd2Mg , Rr2Mg )*/
 _term =  Rd2Mg ;
 _MATELM1( 4 ,4)  += _term;
 _MATELM1( 3 ,4)  -= _term;
 _term =  Rr2Mg ;
 _MATELM1( 4 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
  /* ~ U <-> UMg ( rmc1b , rmc1u )*/
 _term =  rmc1b ;
 _MATELM1( 9 ,9)  += _term;
 _MATELM1( 8 ,9)  -= _term;
 _term =  rmc1u ;
 _MATELM1( 9 ,8)  -= _term;
 _MATELM1( 8 ,8)  += _term;
 /*REACTION*/
  /* ~ Cl <-> ClMg ( rmc2b , rmc2u )*/
 _term =  rmc2b ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 1 ,2)  -= _term;
 _term =  rmc2u ;
 _MATELM1( 2 ,1)  -= _term;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
  /* ~ D1 <-> D1Mg ( rmd1b , rmd1u )*/
 _term =  rmd1b ;
 _MATELM1( 6 ,6)  += _term;
 _MATELM1( 4 ,6)  -= _term;
 _term =  rmd1u ;
 _MATELM1( 6 ,4)  -= _term;
 _MATELM1( 4 ,4)  += _term;
 /*REACTION*/
  /* ~ D2 <-> D2Mg ( rmd2b , rmd2u )*/
 _term =  rmd2b ;
 _MATELM1( 5 ,5)  += _term;
 _MATELM1( 3 ,5)  -= _term;
 _term =  rmd2u ;
 _MATELM1( 5 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
   /* U + Cl + D1 + D2 + O + UMg + ClMg + D1Mg + D2Mg + OMg = 1.0 */
 /*CONSERVATION*/
   } return _reset;
 }
 
/*CVODE end*/
 
static int _ode_count(int _type){ return 10;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
     _ode_spec1 ();
 }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 10; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _cvode_sparse(&_cvsparseobj1, 10, _dlist1, _p, _ode_matsol1, &_coef1);
 }
 
static void _ode_matsol(NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
 _ode_matsol_instance1(_threadargs_);
 }}

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  ClMg = ClMg0;
  Cl = Cl0;
  D2Mg = D2Mg0;
  D1Mg = D1Mg0;
  D2 = D20;
  D1 = D10;
  OMg = OMg0;
  O = O0;
  UMg = UMg0;
  U = U0;
 {
   U = 1.0 ;
   }
  _sav_indep = t; t = _save;

}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
 initmodel();
}}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   g = gmax * O ;
   i = ( 1e-6 ) * g * ( v - Erev ) ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 _g = _nrn_current(_v + .001);
 	{ _rhs = _nrn_current(_v);
 	}
 _g = (_g - _rhs)/.001;
 _g *=  1.e2/(_nd_area);
 _rhs *= 1.e2/(_nd_area);
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
double _dtsav = dt;
if (secondorder) { dt *= 0.5; }
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
 { error = sparse(&_sparseobj1, 10, _slist1, _dlist1, _p, &t, dt, kstates,&_coef1, _linmat1);
 if(error){fprintf(stderr,"at line 165 in file NMDA_Mg_T.mod:\n\n"); nrn_complain(_p); abort_run(error);}
    if (secondorder) {
    int _i;
    for (_i = 0; _i < 10; ++_i) {
      _p[_slist1[_i]] += dt*_p[_dlist1[_i]];
    }}
 }}}
 dt = _dtsav;
}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = OMg_columnindex;  _dlist1[0] = DOMg_columnindex;
 _slist1[1] = ClMg_columnindex;  _dlist1[1] = DClMg_columnindex;
 _slist1[2] = Cl_columnindex;  _dlist1[2] = DCl_columnindex;
 _slist1[3] = D2Mg_columnindex;  _dlist1[3] = DD2Mg_columnindex;
 _slist1[4] = D1Mg_columnindex;  _dlist1[4] = DD1Mg_columnindex;
 _slist1[5] = D2_columnindex;  _dlist1[5] = DD2_columnindex;
 _slist1[6] = D1_columnindex;  _dlist1[6] = DD1_columnindex;
 _slist1[7] = O_columnindex;  _dlist1[7] = DO_columnindex;
 _slist1[8] = UMg_columnindex;  _dlist1[8] = DUMg_columnindex;
 _slist1[9] = U_columnindex;  _dlist1[9] = DU_columnindex;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/wybo/Code/NEAT_public/neat/tools/simtools/neuron/tmp/multichannel_test/mech/NMDA_Mg_T.mod";
static const char* nmodl_file_text = 
  "TITLE kinetic NMDA receptor model\n"
  "\n"
  "COMMENT\n"
  "-----------------------------------------------------------------------------\n"
  "\n"
  "	Kinetic model of NMDA receptors\n"
  "	===============================\n"
  "\n"
  "	10-state gating model:\n"
  "	Kampa et al. (2004) J Physiol\n"
  "\n"
  "	  U -- Cl  --  O\n"
  "         \\   | \\	    \\\n"
  "          \\  |  \\      \\\n"
  "         UMg --  ClMg - OMg\n"
  "		 |	|\n"
  "		D1	|\n"
  "		 | \\	|\n"
  "		D2  \\	|\n"
  "		   \\	D1Mg\n"
  "		    \\	|\n"
  "			D2Mg\n"
  "-----------------------------------------------------------------------------\n"
  "\n"
  "  Based on voltage-clamp recordings of NMDA receptor-mediated currents in\n"
  "  nucleated patches of  rat neocortical layer 5 pyramidal neurons (Kampa 2004),\n"
  "  this model was fit with AxoGraph directly to experimental recordings in\n"
  "  order to obtain the optimal values for the parameters.\n"
  "\n"
  "-----------------------------------------------------------------------------\n"
  "\n"
  "  This mod file does not include mechanisms for the release and time course\n"
  "  of transmitter; it should to be used in conjunction with a sepearate mechanism\n"
  "  to describe the release of transmitter and tiemcourse of the concentration\n"
  "  of transmitter in the synaptic cleft (to be connected to pointer C here).\n"
  "\n"
  "-----------------------------------------------------------------------------\n"
  "\n"
  "  See details of NEURON kinetic models in:\n"
  "\n"
  "  Destexhe, A., Mainen, Z.F. and Sejnowski, T.J.  Kinetic models of\n"
  "  synaptic transmission.  In: Methods in Neuronal Modeling (2nd edition;\n"
  "  edited by Koch, C. and Segev, I.), MIT press, Cambridge, 1996.\n"
  "\n"
  "\n"
  "  Written by Bjoern Kampa in 2004\n"
  "\n"
  "-----------------------------------------------------------------------------\n"
  "\n"
  "  Rates modified for near physiological temperatures with Q10 values from\n"
  "  O.Cais et al 2008, Mg unbinding from Vargas-Caballero 2003, opening and\n"
  "  closing from Lester and Jahr 1992.\n"
  "\n"
  "  Tiago Branco 2010\n"
  "\n"
  "-----------------------------------------------------------------------------\n"
  "\n"
  "ENDCOMMENT\n"
  "\n"
  "\n"
  "INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}\n"
  "\n"
  "NEURON {\n"
  "	POINT_PROCESS NMDA_Mg_T\n"
  "	POINTER C\n"
  "	RANGE U, Cl, D1, D2, O, UMg, ClMg, D1Mg, D2Mg, OMg\n"
  "	RANGE g, gmax, rb, rmb, rmu, rbMg,rmc1b,rmc1u,rmc2b,rmc2u, Erev, mg\n"
  "	GLOBAL Rb, Ru, Rd1, Rr1, Rd2, Rr2, Ro, Rc, Rmb, Rmu\n"
  "	GLOBAL RbMg, RuMg, Rd1Mg, Rr1Mg, Rd2Mg, Rr2Mg, RoMg, RcMg\n"
  "	GLOBAL Rmd1b,Rmd1u,Rmd2b,Rmd2u,rmd1b,rmd1u,rmd2b,rmd2u\n"
  "	GLOBAL Rmc1b,Rmc1u,Rmc2b,Rmc2u\n"
  "	GLOBAL vmin, vmax, valence, memb_fraction\n"
  "	NONSPECIFIC_CURRENT i\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	(nA) = (nanoamp)\n"
  "	(mV) = (millivolt)\n"
  "	(pS) = (picosiemens)\n"
  "	(umho) = (micromho)\n"
  "	(mM) = (milli/liter)\n"
  "	(uM) = (micro/liter)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "\n"
  "	Erev	= 5    	(mV)	: reversal potential\n"
  "	gmax	= 500  	(pS)	: maximal conductance\n"
  "	mg	= 1  	(mM)	: external magnesium concentration\n"
  "	vmin 	= -120	(mV)\n"
  "	vmax 	= 100	(mV)\n"
  "	valence = -2		: parameters of voltage-dependent Mg block\n"
  "	memb_fraction = 0.8\n"
  "\n"
  ": Rates\n"
  "\n"
  "	Rb		= 10e-3    	(/uM /ms)	: binding\n"
  "	Ru		= 0.02016 	(/ms)	: unbinding\n"
  "	Ro		= 46.5e-3   	(/ms)	: opening\n"
  "	Rc		= 91.6e-3   	(/ms)	: closing\n"
  "	Rd1		= 0.02266  	(/ms)	: fast desensitisation\n"
  "	Rr1		= 0.00736  	(/ms)	: fast resensitisation\n"
  "	Rd2 		= 0.004429	(/ms)	: slow desensitisation\n"
  "	Rr2 		= 0.0023	(/ms)	: slow resensitisation\n"
  "	Rmb		= 0.05e-3	(/uM /ms)	: Mg binding Open\n"
  "	Rmu		= 12800e-3	(/ms)	: Mg unbinding Open\n"
  "	Rmc1b		= 0.00005e-3	(/uM /ms)	: Mg binding Closed\n"
  "	Rmc1u		= 0.06	(/ms)	: Mg unbinding Closed\n"
  "	Rmc2b		= 0.00005e-3	(/uM /ms)	: Mg binding Closed2\n"
  "	Rmc2u		= 0.06	(/ms)	: Mg unbinding Closed2\n"
  "	Rmd1b		= 0.00005e-3	(/uM /ms)	: Mg binding Desens1\n"
  "	Rmd1u		= 0.06	(/ms)	: Mg unbinding Desens1\n"
  "	Rmd2b		= 0.00005e-3	(/uM /ms)	: Mg binding Desens2\n"
  "	Rmd2u		= 0.06	(/ms)	: Mg unbinding Desens2\n"
  "	RbMg		= 10e-3		(/uM /ms)	: binding with Mg\n"
  "	RuMg		= 0.06156	(/ms)	: unbinding with Mg\n"
  "	RoMg		= 46.5e-3		(/ms)	: opening with Mg\n"
  "	RcMg		= 91.6e-3	(/ms)	: closing with Mg\n"
  "	Rd1Mg		= 0.02163	(/ms)	: fast desensitisation with Mg\n"
  "	Rr1Mg		= 0.004002	(/ms)	: fast resensitisation with Mg\n"
  "	Rd2Mg		= 0.002678	(/ms)	: slow desensitisation with Mg\n"
  "	Rr2Mg		= 0.001932	(/ms)	: slow resensitisation with Mg\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	v		(mV)	: postsynaptic voltage\n"
  "	i 		(nA)	: current = g*(v - Erev)\n"
  "	g 		(pS)	: conductance\n"
  "	C 		(mM)	: pointer to glutamate concentration\n"
  "\n"
  "	rb		(/ms)   : binding, [glu] dependent\n"
  "	rmb		(/ms)	: blocking V and [Mg] dependent\n"
  "	rmu		(/ms)	: unblocking V and [Mg] dependent\n"
  "	rbMg		(/ms)	: binding, [glu] dependent\n"
  "	rmc1b		(/ms)	: blocking V and [Mg] dependent\n"
  "	rmc1u		(/ms)	: unblocking V and [Mg] dependent\n"
  "	rmc2b		(/ms)	: blocking V and [Mg] dependent\n"
  "	rmc2u		(/ms)	: unblocking V and [Mg] dependent\n"
  "	rmd1b		(/ms)	: blocking V and [Mg] dependent\n"
  "	rmd1u		(/ms)	: unblocking V and [Mg] dependent\n"
  "	rmd2b		(/ms)	: blocking V and [Mg] dependent\n"
  "	rmd2u		(/ms)	: unblocking V and [Mg] dependent\n"
  "}\n"
  "\n"
  "STATE {\n"
  "	: Channel states (all fractions)\n"
  "	U		: unbound\n"
  "	Cl		: closed\n"
  "	D1		: desensitised 1\n"
  "	D2		: desensitised 2\n"
  "	O		: open\n"
  "	UMg		: unbound with Mg\n"
  "	ClMg		: closed with Mg\n"
  "	D1Mg		: desensitised 1 with Mg\n"
  "	D2Mg		: desensitised 2 with Mg\n"
  "	OMg		: open with Mg\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "	U = 1\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE kstates METHOD sparse\n"
  "\n"
  "	g = gmax * O\n"
  "	i = (1e-6) * g * (v - Erev)\n"
  "}\n"
  "\n"
  "KINETIC kstates {\n"
  "\n"
  "	rb 	= Rb 	* (1e3) * C\n"
  "	rbMg 	= RbMg 	* (1e3) * C\n"
  "	rmb 	= Rmb 	* mg * (1e3) * exp((v-40) * valence * memb_fraction /25)\n"
  "	rmu 	= Rmu 	* exp((-1)*(v-40) * valence * (1-memb_fraction) /25)\n"
  "	rmc1b 	= Rmc1b * mg * (1e3) * exp((v-40) * valence * memb_fraction /25)\n"
  "	rmc1u 	= Rmc1u * exp((-1)*(v-40) * valence * (1-memb_fraction) /25)\n"
  "	rmc2b 	= Rmc2b * mg * (1e3) * exp((v-40) * valence * memb_fraction /25)\n"
  "	rmc2u 	= Rmc2u * exp((-1)*(v-40) * valence * (1-memb_fraction) /25)\n"
  "	rmd1b 	= Rmd1b * mg * (1e3) * exp((v-40) * valence * memb_fraction /25)\n"
  "	rmd1u 	= Rmd1u * exp((-1)*(v-40) * valence * (1-memb_fraction) /25)\n"
  "	rmd2b 	= Rmd2b * mg * (1e3) * exp((v-40) * valence * memb_fraction /25)\n"
  "	rmd2u 	= Rmd2u * exp((-1)*(v-40) * valence * (1-memb_fraction) /25)\n"
  "\n"
  "	~ U <-> Cl	(rb,Ru)\n"
  "	~ Cl <-> O	(Ro,Rc)\n"
  "	~ Cl <-> D1	(Rd1,Rr1)\n"
  "	~ D1 <-> D2	(Rd2,Rr2)\n"
  "	~ O <-> OMg	(rmb,rmu)\n"
  "	~ UMg <-> ClMg 	(rbMg,RuMg)\n"
  "	~ ClMg <-> OMg 	(RoMg,RcMg)\n"
  "	~ ClMg <-> D1Mg (Rd1Mg,Rr1Mg)\n"
  "	~ D1Mg <-> D2Mg (Rd2Mg,Rr2Mg)\n"
  "	~ U <-> UMg     (rmc1b,rmc1u)\n"
  "	~ Cl <-> ClMg	(rmc2b,rmc2u)\n"
  "	~ D1 <-> D1Mg	(rmd1b,rmd1u)\n"
  "	~ D2 <-> D2Mg	(rmd2b,rmd2u)\n"
  "\n"
  "	CONSERVE U+Cl+D1+D2+O+UMg+ClMg+D1Mg+D2Mg+OMg = 1\n"
  "}\n"
  "\n"
  ;
#endif
