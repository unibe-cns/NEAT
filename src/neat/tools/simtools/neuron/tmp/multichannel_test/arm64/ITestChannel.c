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
 
#define nrn_init _nrn_init__ITestChannel
#define _nrn_initial _nrn_initial__ITestChannel
#define nrn_cur _nrn_cur__ITestChannel
#define _nrn_current _nrn_current__ITestChannel
#define nrn_jacob _nrn_jacob__ITestChannel
#define nrn_state _nrn_state__ITestChannel
#define _net_receive _net_receive__ITestChannel 
#define rates rates__ITestChannel 
#define states states__ITestChannel 
 
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
#define g _p[0]
#define g_columnindex 0
#define e _p[1]
#define e_columnindex 1
#define i _p[2]
#define i_columnindex 2
#define a00 _p[3]
#define a00_columnindex 3
#define a01 _p[4]
#define a01_columnindex 4
#define a02 _p[5]
#define a02_columnindex 5
#define a10 _p[6]
#define a10_columnindex 6
#define a11 _p[7]
#define a11_columnindex 7
#define a12 _p[8]
#define a12_columnindex 8
#define temp _p[9]
#define temp_columnindex 9
#define Da00 _p[10]
#define Da00_columnindex 10
#define Da01 _p[11]
#define Da01_columnindex 11
#define Da02 _p[12]
#define Da02_columnindex 12
#define Da10 _p[13]
#define Da10_columnindex 13
#define Da11 _p[14]
#define Da11_columnindex 14
#define Da12 _p[15]
#define Da12_columnindex 15
#define v _p[16]
#define v_columnindex 16
#define _g _p[17]
#define _g_columnindex 17
 
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
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_rates(void);
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

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_ITestChannel", _hoc_setdata,
 "rates_ITestChannel", _hoc_rates,
 0, 0
};
 /* declare global and static user variables */
 static int _thread1data_inuse = 0;
static double _thread1data[12];
#define _gth 0
#define a12_inf_ITestChannel _thread1data[0]
#define a12_inf _thread[_gth]._pval[0]
#define a11_inf_ITestChannel _thread1data[1]
#define a11_inf _thread[_gth]._pval[1]
#define a10_inf_ITestChannel _thread1data[2]
#define a10_inf _thread[_gth]._pval[2]
#define a02_inf_ITestChannel _thread1data[3]
#define a02_inf _thread[_gth]._pval[3]
#define a01_inf_ITestChannel _thread1data[4]
#define a01_inf _thread[_gth]._pval[4]
#define a00_inf_ITestChannel _thread1data[5]
#define a00_inf _thread[_gth]._pval[5]
#define tau_a12_ITestChannel _thread1data[6]
#define tau_a12 _thread[_gth]._pval[6]
#define tau_a11_ITestChannel _thread1data[7]
#define tau_a11 _thread[_gth]._pval[7]
#define tau_a10_ITestChannel _thread1data[8]
#define tau_a10 _thread[_gth]._pval[8]
#define tau_a02_ITestChannel _thread1data[9]
#define tau_a02 _thread[_gth]._pval[9]
#define tau_a01_ITestChannel _thread1data[10]
#define tau_a01 _thread[_gth]._pval[10]
#define tau_a00_ITestChannel _thread1data[11]
#define tau_a00 _thread[_gth]._pval[11]
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "tau_a00_ITestChannel", "ms",
 "tau_a01_ITestChannel", "ms",
 "tau_a02_ITestChannel", "ms",
 "tau_a10_ITestChannel", "ms",
 "tau_a11_ITestChannel", "ms",
 "tau_a12_ITestChannel", "ms",
 "g_ITestChannel", "S/cm2",
 "e_ITestChannel", "mV",
 "i_ITestChannel", "mA/cm2",
 0,0
};
 static double a120 = 0;
 static double a110 = 0;
 static double a100 = 0;
 static double a020 = 0;
 static double a010 = 0;
 static double a000 = 0;
 static double delta_t = 0.01;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "a00_inf_ITestChannel", &a00_inf_ITestChannel,
 "tau_a00_ITestChannel", &tau_a00_ITestChannel,
 "a01_inf_ITestChannel", &a01_inf_ITestChannel,
 "tau_a01_ITestChannel", &tau_a01_ITestChannel,
 "a02_inf_ITestChannel", &a02_inf_ITestChannel,
 "tau_a02_ITestChannel", &tau_a02_ITestChannel,
 "a10_inf_ITestChannel", &a10_inf_ITestChannel,
 "tau_a10_ITestChannel", &tau_a10_ITestChannel,
 "a11_inf_ITestChannel", &a11_inf_ITestChannel,
 "tau_a11_ITestChannel", &tau_a11_ITestChannel,
 "a12_inf_ITestChannel", &a12_inf_ITestChannel,
 "tau_a12_ITestChannel", &tau_a12_ITestChannel,
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
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[0]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"ITestChannel",
 "g_ITestChannel",
 "e_ITestChannel",
 0,
 "i_ITestChannel",
 0,
 "a00_ITestChannel",
 "a01_ITestChannel",
 "a02_ITestChannel",
 "a10_ITestChannel",
 "a11_ITestChannel",
 "a12_ITestChannel",
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 18, _prop);
 	/*initialize range parameters*/
 	g = 0;
 	e = -23;
 	_prop->param = _p;
 	_prop->param_size = 18;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 1, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _thread_mem_init(Datum*);
 static void _thread_cleanup(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _ITestChannel_reg() {
	int _vectorized = 1;
  _initlists();
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 2);
  _extcall_thread = (Datum*)ecalloc(1, sizeof(Datum));
  _thread_mem_init(_extcall_thread);
  _thread1data_inuse = 0;
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 1, _thread_mem_init);
     _nrn_thread_reg(_mechtype, 0, _thread_cleanup);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 18, 1);
  hoc_register_dparam_semantics(_mechtype, 0, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 ITestChannel /Users/wybo/Code/NEAT_public/neat/tools/simtools/neuron/tmp/multichannel_test/mech/ITestChannel.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[6], _dlist1[6];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   rates ( _threadargscomma_ v ) ;
   Da00 = ( a00_inf - a00 ) / tau_a00 ;
   Da01 = ( a01_inf - a01 ) / tau_a01 ;
   Da02 = ( a02_inf - a02 ) / tau_a02 ;
   Da10 = ( a10_inf - a10 ) / tau_a10 ;
   Da11 = ( a11_inf - a11 ) / tau_a11 ;
   Da12 = ( a12_inf - a12 ) / tau_a12 ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 rates ( _threadargscomma_ v ) ;
 Da00 = Da00  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau_a00 )) ;
 Da01 = Da01  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau_a01 )) ;
 Da02 = Da02  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau_a02 )) ;
 Da10 = Da10  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau_a10 )) ;
 Da11 = Da11  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau_a11 )) ;
 Da12 = Da12  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau_a12 )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   rates ( _threadargscomma_ v ) ;
    a00 = a00 + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau_a00)))*(- ( ( ( a00_inf ) ) / tau_a00 ) / ( ( ( ( - 1.0 ) ) ) / tau_a00 ) - a00) ;
    a01 = a01 + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau_a01)))*(- ( ( ( a01_inf ) ) / tau_a01 ) / ( ( ( ( - 1.0 ) ) ) / tau_a01 ) - a01) ;
    a02 = a02 + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau_a02)))*(- ( ( ( a02_inf ) ) / tau_a02 ) / ( ( ( ( - 1.0 ) ) ) / tau_a02 ) - a02) ;
    a10 = a10 + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau_a10)))*(- ( ( ( a10_inf ) ) / tau_a10 ) / ( ( ( ( - 1.0 ) ) ) / tau_a10 ) - a10) ;
    a11 = a11 + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau_a11)))*(- ( ( ( a11_inf ) ) / tau_a11 ) / ( ( ( ( - 1.0 ) ) ) / tau_a11 ) - a11) ;
    a12 = a12 + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau_a12)))*(- ( ( ( a12_inf ) ) / tau_a12 ) / ( ( ( ( - 1.0 ) ) ) / tau_a12 ) - a12) ;
   }
  return 0;
}
 
static int  rates ( _threadargsprotocomma_ double _lv ) {
   temp = celsius ;
   a00_inf = 1.0 / ( 0.74081822068171788 * exp ( 0.01 * _lv ) + 1.0 ) ;
   tau_a00 = 1.0 ;
   a01_inf = 1.0 * exp ( 0.01 * _lv ) / ( 1.0 * exp ( 0.01 * _lv ) + 1.3498588075760032 ) ;
   tau_a01 = 2.0 ;
   a02_inf = - 10.0 ;
   tau_a02 = 1.0 ;
   a10_inf = 2.0 / ( 0.74081822068171788 * exp ( 0.01 * _lv ) + 1.0 ) ;
   tau_a10 = 2.0 ;
   a11_inf = 2.0 * exp ( 0.01 * _lv ) / ( 1.0 * exp ( 0.01 * _lv ) + 1.3498588075760032 ) ;
   tau_a11 = 2.0 ;
   a12_inf = - 30.0 ;
   tau_a12 = 3.0 ;
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 rates ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 6;}
 
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
	for (_i=0; _i < 6; ++_i) {
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
 
static void _thread_mem_init(Datum* _thread) {
  if (_thread1data_inuse) {_thread[_gth]._pval = (double*)ecalloc(12, sizeof(double));
 }else{
 _thread[_gth]._pval = _thread1data; _thread1data_inuse = 1;
 }
 }
 
static void _thread_cleanup(Datum* _thread) {
  if (_thread[_gth]._pval == _thread1data) {
   _thread1data_inuse = 0;
  }else{
   free((void*)_thread[_gth]._pval);
  }
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  a12 = a120;
  a11 = a110;
  a10 = a100;
  a02 = a020;
  a01 = a010;
  a00 = a000;
 {
   rates ( _threadargscomma_ v ) ;
   a00 = a00_inf ;
   a01 = a01_inf ;
   a02 = a02_inf ;
   a10 = a10_inf ;
   a11 = a11_inf ;
   a12 = a12_inf ;
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
   i = g * ( 5.0 * pow ( a00 , 3.0 ) * pow ( a01 , 3.0 ) * a02 + pow ( a10 , 2.0 ) * pow ( a11 , 2.0 ) * a12 ) * ( v - e ) ;
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
 {   states(_p, _ppvar, _thread, _nt);
  }}}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = a00_columnindex;  _dlist1[0] = Da00_columnindex;
 _slist1[1] = a01_columnindex;  _dlist1[1] = Da01_columnindex;
 _slist1[2] = a02_columnindex;  _dlist1[2] = Da02_columnindex;
 _slist1[3] = a10_columnindex;  _dlist1[3] = Da10_columnindex;
 _slist1[4] = a11_columnindex;  _dlist1[4] = Da11_columnindex;
 _slist1[5] = a12_columnindex;  _dlist1[5] = Da12_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/wybo/Code/NEAT_public/neat/tools/simtools/neuron/tmp/multichannel_test/mech/ITestChannel.mod";
static const char* nmodl_file_text = 
  ": This mod file is automaticaly generated by the ``neat.channels.ionchannels`` module\n"
  "\n"
  "NEURON {\n"
  "    SUFFIX ITestChannel\n"
  "    NONSPECIFIC_CURRENT i\n"
  "    RANGE  g, e\n"
  "    GLOBAL a00_inf, a01_inf, a02_inf, a10_inf, a11_inf, a12_inf, tau_a00, tau_a01, tau_a02, tau_a10, tau_a11, tau_a12\n"
  "    THREADSAFE\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "    g = 0.0 (S/cm2)\n"
  "    e = -23.0 (mV)\n"
  "    celsius (degC)\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "    (mA) = (milliamp)\n"
  "    (mV) = (millivolt)\n"
  "    (mM) = (milli/liter)\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "    i (mA/cm2)\n"
  "    a00_inf      \n"
  "    tau_a00 (ms) \n"
  "    a01_inf      \n"
  "    tau_a01 (ms) \n"
  "    a02_inf      \n"
  "    tau_a02 (ms) \n"
  "    a10_inf      \n"
  "    tau_a10 (ms) \n"
  "    a11_inf      \n"
  "    tau_a11 (ms) \n"
  "    a12_inf      \n"
  "    tau_a12 (ms) \n"
  "    v (mV)\n"
  "    temp (degC)\n"
  "}\n"
  "\n"
  "STATE {\n"
  "    a00\n"
  "    a01\n"
  "    a02\n"
  "    a10\n"
  "    a11\n"
  "    a12\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "    SOLVE states METHOD cnexp\n"
  "    i = g * (5*pow(a00, 3)*pow(a01, 3)*a02 + pow(a10, 2)*pow(a11, 2)*a12) * (v - e)\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "    rates(v)\n"
  "    a00 = a00_inf\n"
  "    a01 = a01_inf\n"
  "    a02 = a02_inf\n"
  "    a10 = a10_inf\n"
  "    a11 = a11_inf\n"
  "    a12 = a12_inf\n"
  "}\n"
  "\n"
  "DERIVATIVE states {\n"
  "    rates(v)\n"
  "    a00' = (a00_inf - a00) /  tau_a00 \n"
  "    a01' = (a01_inf - a01) /  tau_a01 \n"
  "    a02' = (a02_inf - a02) /  tau_a02 \n"
  "    a10' = (a10_inf - a10) /  tau_a10 \n"
  "    a11' = (a11_inf - a11) /  tau_a11 \n"
  "    a12' = (a12_inf - a12) /  tau_a12 \n"
  "}\n"
  "\n"
  "PROCEDURE rates(v) {\n"
  "    temp = celsius\n"
  "    a00_inf = 1.0/(0.74081822068171788*exp(0.01*v) + 1.0)\n"
  "    tau_a00 = 1.0\n"
  "    a01_inf = 1.0*exp(0.01*v)/(1.0*exp(0.01*v) + 1.3498588075760032)\n"
  "    tau_a01 = 2.0\n"
  "    a02_inf = -10.0\n"
  "    tau_a02 = 1.0\n"
  "    a10_inf = 2.0/(0.74081822068171788*exp(0.01*v) + 1.0)\n"
  "    tau_a10 = 2.0\n"
  "    a11_inf = 2.0*exp(0.01*v)/(1.0*exp(0.01*v) + 1.3498588075760032)\n"
  "    tau_a11 = 2.0\n"
  "    a12_inf = -30.0\n"
  "    tau_a12 = 3.0\n"
  "}\n"
  "\n"
  ;
#endif
