/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
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
 
#define nrn_init _nrn_init__exp_AMPA_NMDA
#define _nrn_initial _nrn_initial__exp_AMPA_NMDA
#define nrn_cur _nrn_cur__exp_AMPA_NMDA
#define _nrn_current _nrn_current__exp_AMPA_NMDA
#define nrn_jacob _nrn_jacob__exp_AMPA_NMDA
#define nrn_state _nrn_state__exp_AMPA_NMDA
#define _net_receive _net_receive__exp_AMPA_NMDA 
#define _f_mgblock _f_mgblock__exp_AMPA_NMDA 
#define betadyn betadyn__exp_AMPA_NMDA 
#define mgblock mgblock__exp_AMPA_NMDA 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define tau _p[0]
#define tau_columnindex 0
#define tau_NMDA _p[1]
#define tau_NMDA_columnindex 1
#define e _p[2]
#define e_columnindex 2
#define mg _p[3]
#define mg_columnindex 3
#define NMDA_ratio _p[4]
#define NMDA_ratio_columnindex 4
#define i _p[5]
#define i_columnindex 5
#define g _p[6]
#define g_columnindex 6
#define g_NMDA _p[7]
#define g_NMDA_columnindex 7
#define B _p[8]
#define B_columnindex 8
#define Dg _p[9]
#define Dg_columnindex 9
#define Dg_NMDA _p[10]
#define Dg_NMDA_columnindex 10
#define v _p[11]
#define v_columnindex 11
#define _g _p[12]
#define _g_columnindex 12
#define _tsav _p[13]
#define _tsav_columnindex 13
#define _nd_area  *_ppvar[0]._pval
 
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
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_mgblock(void*);
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
 _extcall_prop = _prop;
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
 "mgblock", _hoc_mgblock,
 0, 0
};
 
static void _check_mgblock(double*, Datum*, Datum*, NrnThread*); 
static void _check_table_thread(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, int _type) {
   _check_mgblock(_p, _ppvar, _thread, _nt);
 }
 /* declare global and static user variables */
#define usetable usetable_exp_AMPA_NMDA
 double usetable = 1;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "usetable_exp_AMPA_NMDA", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "tau", "ms",
 "tau_NMDA", "ms",
 "e", "mV",
 "mg", "mM",
 "g", "uS",
 "g_NMDA", "uS",
 "i", "nA",
 0,0
};
 static double delta_t = 0.01;
 static double g_NMDA0 = 0;
 static double g0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "usetable_exp_AMPA_NMDA", &usetable_exp_AMPA_NMDA,
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
 
#define _cvode_ieq _ppvar[2]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"exp_AMPA_NMDA",
 "tau",
 "tau_NMDA",
 "e",
 "mg",
 "NMDA_ratio",
 0,
 "i",
 0,
 "g",
 "g_NMDA",
 0,
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
 	_p = nrn_prop_data_alloc(_mechtype, 14, _prop);
 	/*initialize range parameters*/
 	tau = 10;
 	tau_NMDA = 200;
 	e = 0;
 	mg = 1;
 	NMDA_ratio = 0;
  }
 	_prop->param = _p;
 	_prop->param_size = 14;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 3, _prop);
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
 static void _net_receive(Point_process*, double*, double);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _exp_AMPA_NMDA_reg() {
	int _vectorized = 1;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 1,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_table_reg(_mechtype, _check_table_thread);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 14, 3);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 1;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 exp_AMPA_NMDA /Users/wybo/Code/NEAT_public/neat/tools/simtools/neuron/tmp/multichannel_test/mech/exp_AMPA_NMDA.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double *_t_B;
static int _reset;
static char *modelname = "NMDA synapse for nucleus accumbens model";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int _f_mgblock(_threadargsprotocomma_ double);
static int mgblock(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static void _n_mgblock(_threadargsprotocomma_ double _lv);
 static int _slist1[2], _dlist1[2];
 static int betadyn(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   Dg = - g / tau ;
   Dg_NMDA = - g_NMDA / tau_NMDA ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 Dg = Dg  / (1. - dt*( ( - 1.0 ) / tau )) ;
 Dg_NMDA = Dg_NMDA  / (1. - dt*( ( - 1.0 ) / tau_NMDA )) ;
  return 0;
}
 /*END CVODE*/
 static int betadyn (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
    g = g + (1. - exp(dt*(( - 1.0 ) / tau)))*(- ( 0.0 ) / ( ( - 1.0 ) / tau ) - g) ;
    g_NMDA = g_NMDA + (1. - exp(dt*(( - 1.0 ) / tau_NMDA)))*(- ( 0.0 ) / ( ( - 1.0 ) / tau_NMDA ) - g_NMDA) ;
   }
  return 0;
}
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{  double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _thread = (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;   _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t; {
     if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = g;
    double __primary = (g + _args[0]) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - 1.0 ) / tau ) ) )*( - ( 0.0 ) / ( ( - 1.0 ) / tau ) - __primary );
    g += __primary;
  } else {
 g = g + _args[0] ;
     }
   if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = g_NMDA;
    double __primary = (g_NMDA + _args[0] * NMDA_ratio) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - 1.0 ) / tau_NMDA ) ) )*( - ( 0.0 ) / ( ( - 1.0 ) / tau_NMDA ) - __primary );
    g_NMDA += __primary;
  } else {
 g_NMDA = g_NMDA + _args[0] * NMDA_ratio ;
     }
 } }
 static double _mfac_mgblock, _tmin_mgblock;
  static void _check_mgblock(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  static double _sav_mg;
  if (!usetable) {return;}
  if (_sav_mg != mg) { _maktable = 1;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_mgblock =  - 100.0 ;
   _tmax =  100.0 ;
   _dx = (_tmax - _tmin_mgblock)/201.; _mfac_mgblock = 1./_dx;
   for (_i=0, _x=_tmin_mgblock; _i < 202; _x += _dx, _i++) {
    _f_mgblock(_p, _ppvar, _thread, _nt, _x);
    _t_B[_i] = B;
   }
   _sav_mg = mg;
  }
 }

 static int mgblock(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lv) { 
#if 0
_check_mgblock(_p, _ppvar, _thread, _nt);
#endif
 _n_mgblock(_p, _ppvar, _thread, _nt, _lv);
 return 0;
 }

 static void _n_mgblock(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 _f_mgblock(_p, _ppvar, _thread, _nt, _lv); return; 
}
 _xi = _mfac_mgblock * (_lv - _tmin_mgblock);
 if (isnan(_xi)) {
  B = _xi;
  return;
 }
 if (_xi <= 0.) {
 B = _t_B[0];
 return; }
 if (_xi >= 201.) {
 B = _t_B[201];
 return; }
 _i = (int) _xi;
 _theta = _xi - (double)_i;
 B = _t_B[_i] + _theta*(_t_B[_i+1] - _t_B[_i]);
 }

 
static int  _f_mgblock ( _threadargsprotocomma_ double _lv ) {
   B = 1.0 / ( 1.0 + exp ( 0.062 * - _lv ) * ( mg / 3.57 ) ) ;
    return 0; }
 
static double _hoc_mgblock(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 
#if 1
 _check_mgblock(_p, _ppvar, _thread, _nt);
#endif
 _r = 1.;
 mgblock ( _p, _ppvar, _thread, _nt, *getarg(1) );
 return(_r);
}
 
static int _ode_count(int _type){ return 2;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
     _ode_spec1 (_p, _ppvar, _thread, _nt);
 }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 2; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
 _ode_matsol_instance1(_threadargs_);
 }}

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  g_NMDA = g_NMDA0;
  g = g0;
 {
   g = 0.0 ;
   }
 
}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];

#if 0
 _check_mgblock(_p, _ppvar, _thread, _nt);
#endif
 _tsav = -1e20;
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
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   mgblock ( _threadargscomma_ v ) ;
   i = ( g + g_NMDA * B ) * ( v - e ) ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
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
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
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
 
}
 
}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
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
 
}
 
}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
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
 {   betadyn(_p, _ppvar, _thread, _nt);
  }}}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = g_columnindex;  _dlist1[0] = Dg_columnindex;
 _slist1[1] = g_NMDA_columnindex;  _dlist1[1] = Dg_NMDA_columnindex;
   _t_B = makevector(202*sizeof(double));
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/wybo/Code/NEAT_public/neat/tools/simtools/neuron/tmp/multichannel_test/mech/exp_AMPA_NMDA.mod";
static const char* nmodl_file_text = 
  "TITLE   NMDA synapse for nucleus accumbens model\n"
  ": see comments below\n"
  "\n"
  "NEURON {\n"
  "	POINT_PROCESS exp_AMPA_NMDA\n"
  "	RANGE tau, tau_NMDA, mg, i, e, NMDA_ratio\n"
  "	NONSPECIFIC_CURRENT i\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	(nA) = (nanoamp)\n"
  "	(mV) = (millivolt)\n"
  "	(uS) = (microsiemens)\n"
  "	(mM) = (milli/liter)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	tau = 10 (ms)\n"
  "    tau_NMDA = 200 (ms)\n"
  "	e  = 0    (mV)   : reversal potential, Dalby 2003\n"
  "	mg = 1      (mM)    : external magnesium concentration\n"
  "    NMDA_ratio = 0.0\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	v (mV)   		: postsynaptic voltage\n"
  "	i (nA)   		: nonspecific current = g*(v - Erev)\n"
  "\n"
  "	B				: voltage dependendent magnesium blockade\n"
  "}\n"
  "\n"
  "\n"
  "STATE { \n"
  "	g (uS)\n"
  "    g_NMDA (uS)\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "    g = 0\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE betadyn METHOD cnexp\n"
  "  	mgblock(v)\n"
  "	i = (g + g_NMDA * B) * (v - e)	\n"
  "}\n"
  "\n"
  "DERIVATIVE betadyn {\n"
  "    g' = -g/tau\n"
  "    g_NMDA' = -g_NMDA/tau_NMDA\n"
  "}\n"
  "\n"
  "NET_RECEIVE(weight (uS)) {\n"
  "    g = g + weight\n"
  "    g_NMDA = g_NMDA + weight * NMDA_ratio\n"
  "}\n"
  "\n"
  "\n"
  "PROCEDURE mgblock( v(mV) ) {\n"
  "	: from Jahr & Stevens\n"
  "\n"
  "	TABLE B DEPEND mg\n"
  "		FROM -100 TO 100 WITH 201\n"
  "\n"
  "	B = 1 / (1 + exp(0.062 (/mV) * -v) * (mg / 3.57 (mM)))\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "COMMENT\n"
  "Author Johan Hake (c) spring 2004\n"
  ":     Summate input from many presynaptic sources and saturate \n"
  ":     each one of them during heavy presynaptic firing\n"
  "\n"
  ": [1] Destexhe, A., Z. F. Mainen and T. J. Sejnowski (1998)\n"
  ":     Kinetic models of synaptic transmission\n"
  ":     In C. Koch and I. Segev (Eds.), Methods in Neuronal Modeling\n"
  "\n"
  ": [2] Rotter, S. and M. Diesmann (1999) Biol. Cybern. 81, 381-402\n"
  ":     Exact digital simulation of time-invariant linear systems with application \n"
  ":     to neural modeling\n"
  "\n"
  "Mainen ZF, Malinow R, Svoboda K (1999) Nature. 399, 151-155.\n"
  "Synaptic calcium transients in single spines indicate that NMDA\n"
  "receptors are not saturated.\n"
  "\n"
  "Chapman DE, Keefe KA, Wilcox KS (2003) J Neurophys. 89: 69-80.\n"
  "Evidence for functionally distinct synaptic nmda receptors in ventromedial\n"
  "vs. dorsolateral striatum.\n"
  "\n"
  "Dalby, N. O., and Mody, I. (2003). Activation of NMDA receptors in rat\n"
  "dentate gyrus granule cells by spontaneous and evoked transmitter\n"
  "release. J Neurophysiol 90, 786-797.\n"
  "\n"
  "Jahr CE, Stevens CF. (1990) Voltage dependence of NMDA activated\n"
  "macroscopic conductances predicted by single channel kinetics. J\n"
  "Neurosci 10: 3178, 1990.\n"
  "\n"
  "Gutfreund H, Kinetics for the Life Sciences, Cambridge University Press, 1995, pg 234.\n"
  "(suggested by Ted Carnevale)\n"
  "ENDCOMMENT\n"
  ;
#endif
