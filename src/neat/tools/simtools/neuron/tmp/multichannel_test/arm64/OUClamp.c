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
 
#define nrn_init _nrn_init__OUClamp
#define _nrn_initial _nrn_initial__OUClamp
#define nrn_cur _nrn_cur__OUClamp
#define _nrn_current _nrn_current__OUClamp
#define nrn_jacob _nrn_jacob__OUClamp
#define nrn_state _nrn_state__OUClamp
#define _net_receive _net_receive__OUClamp 
#define oup oup__OUClamp 
#define seed seed__OUClamp 
 
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
#define delay _p[0]
#define delay_columnindex 0
#define dur _p[1]
#define dur_columnindex 1
#define tau _p[2]
#define tau_columnindex 2
#define mean _p[3]
#define mean_columnindex 3
#define stdev _p[4]
#define stdev_columnindex 4
#define seed_usr _p[5]
#define seed_usr_columnindex 5
#define dt_usr _p[6]
#define dt_usr_columnindex 6
#define i _p[7]
#define i_columnindex 7
#define on _p[8]
#define on_columnindex 8
#define per _p[9]
#define per_columnindex 9
#define ival _p[10]
#define ival_columnindex 10
#define ivar _p[11]
#define ivar_columnindex 11
#define flag1 _p[12]
#define flag1_columnindex 12
#define exp_decay _p[13]
#define exp_decay_columnindex 13
#define amp_gauss _p[14]
#define amp_gauss_columnindex 14
#define donotuse _p[15]
#define donotuse_columnindex 15
#define _g _p[16]
#define _g_columnindex 16
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
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_oup(void*);
 static double _hoc_seed(void*);
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
 "oup", _hoc_oup,
 "seed", _hoc_seed,
 0, 0
};
 /* declare global and static user variables */
#define noc noc_OUClamp
 double noc = 0;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "dur", 0, 1e+09,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "delay", "ms",
 "dur", "ms",
 "tau", "ms",
 "mean", "nA",
 "stdev", "nA",
 "seed_usr", "1",
 "dt_usr", "ms",
 "i", "nA",
 0,0
};
 static double delta_t = 0.01;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "noc_OUClamp", &noc_OUClamp,
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
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"OUClamp",
 "delay",
 "dur",
 "tau",
 "mean",
 "stdev",
 "seed_usr",
 "dt_usr",
 0,
 "i",
 0,
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
 	_p = nrn_prop_data_alloc(_mechtype, 17, _prop);
 	/*initialize range parameters*/
 	delay = 0;
 	dur = 0;
 	tau = 100;
 	mean = 0;
 	stdev = 1;
 	seed_usr = 42;
 	dt_usr = 0.1;
  }
 	_prop->param = _p;
 	_prop->param_size = 17;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 2, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _OUClamp_reg() {
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
  hoc_register_prop_size(_mechtype, 17, 2);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
 	hoc_register_cvode(_mechtype, _ode_count, 0, 0, 0);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 OUClamp /Users/wybo/Code/NEAT_public/neat/tools/simtools/neuron/tmp/multichannel_test/mech/OUClamp.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int oup();
static int seed(double);
 
static int  seed (  double _lx ) {
   set_seed ( _lx ) ;
    return 0; }
 
static double _hoc_seed(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r = 1.;
 seed (  *getarg(1) );
 return(_r);
}
 
static int  oup (  ) {
   noc = noc + 1.0 ;
   if ( t < delay ) {
     ivar = 0. ;
     }
   else {
     if ( flag1  == 0.0 ) {
       flag1 = 1.0 ;
       ivar = mean ;
       }
     if ( t < delay + dur ) {
       ivar = mean + exp_decay * ( ivar - mean ) + amp_gauss * normrand ( 0.0 , 1.0 ) ;
       }
     else {
       ivar = 0. ;
       }
     }
    return 0; }
 
static double _hoc_oup(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r = 1.;
 oup (  );
 return(_r);
}
 
static int _ode_count(int _type){ hoc_execerror("OUClamp", "cannot be used with CVODE"); return 0;}

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
 {
   on = 0.0 ;
   ivar = 0.0 ;
   i = 0.0 ;
   flag1 = 0.0 ;
   exp_decay = exp ( - dt_usr / tau ) ;
   amp_gauss = stdev * sqrt ( 1. - exp ( - 2. * dt_usr / tau ) ) ;
   seed ( _threadargscomma_ seed_usr ) ;
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
   i = - ivar ;
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
 { error =  oup();
 if(error){fprintf(stderr,"at line 81 in file OUClamp.mod:\n    SOLVE oup\n"); nrn_complain(_p); abort_run(error);}
 }}}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/wybo/Code/NEAT_public/neat/tools/simtools/neuron/tmp/multichannel_test/mech/OUClamp.mod";
static const char* nmodl_file_text = 
  "COMMENT\n"
  "Noise current characterized by gaussian distribution \n"
  "with mean mean and standerd deviation stdev.\n"
  "\n"
  "Borrows from NetStim's code so it can be linked with an external instance \n"
  "of the Random class in order to generate output that is independent of \n"
  "other instances of InGau.\n"
  "\n"
  "User specifies the time at which the noise starts, \n"
  "and the duration of the noise.\n"
  "Since a new value is drawn at each time step, \n"
  "should be used only with fixed time step integration.\n"
  "ENDCOMMENT\n"
  "\n"
  "NEURON {\n"
  "    POINT_PROCESS OUClamp\n"
  "    NONSPECIFIC_CURRENT i\n"
  "    RANGE mean, stdev, tau\n"
  "    RANGE dt_usr\n"
  "    RANGE delay, dur, seed_usr\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "    (nA) = (nanoamp)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "    delay (ms) : delay until noise starts\n"
  "    dur (ms) <0, 1e9> : duration of noise\n"
  "    tau = 100.(ms)\n"
  "    mean = 0 (nA)\n"
  "    stdev = 1 (nA)\n"
  "    seed_usr = 42 (1)\n"
  "    dt_usr = .1 (ms)\n"
  "    noc = 0 \n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "    :dt (ms)\n"
  "    on\n"
  "    per (ms)\n"
  "    ival (nA)\n"
  "    ivar (nA)\n"
  "    i (nA)\n"
  "    flag1\n"
  "    exp_decay\n"
  "    amp_gauss       (nA)\n"
  "    donotuse\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "    :VERBATIM\n"
  "    :  printf(\"dt = %.2f\\n\", dt);\n"
  "    :  printf(\"dt_usr = %.2f\\n\", dt_usr);\n"
  "    :ENDVERBATIM\n"
  "    :per = dt\n"
  "    on = 0\n"
  "    ivar = 0\n"
  "    i = 0\n"
  "    flag1 = 0\n"
  "    exp_decay = exp(-dt_usr/tau) : exp(-dt/tau)\n"
  "    amp_gauss = stdev * sqrt(1. - exp(-2.*dt_usr/tau)) : stdev * sqrt(1. - exp(-2.*dt/tau))\n"
  "    :VERBATIM\n"
  "    :  printf(\"std = %.10f\\n\", amp_gauss);\n"
  "    :  printf(\"std_ = %.10f\\n\", stdev * sqrt(1. - exp(-2.*.1/tau)));\n"
  "    :ENDVERBATIM\n"
  "    seed(seed_usr)\n"
  "}\n"
  "\n"
  "PROCEDURE seed(x) {\n"
  "    set_seed(x)\n"
  "}\n"
  "\n"
  "COMMENT\n"
  "BEFORE BREAKPOINT {\n"
  "    i = -ivar\n"
  "}\n"
  "ENDCOMMENT\n"
  "\n"
  "BREAKPOINT {\n"
  "    SOLVE oup\n"
  "    i = - ivar\n"
  "}\n"
  "\n"
  "PROCEDURE oup() {\n"
  "    noc = noc + 1\n"
  "    :if (t < 2.) {\n"
  "    :    VERBATIM\n"
  "    :        printf(\">>> <<<\\n\");\n"
  "    :        printf(\"noc = %.2f\\n\", noc);\n"
  "    :        printf(\"t  = %.2f\\n\", t);\n"
  "    :        printf(\"dt = %.2f\\n\", dt);\n"
  "    :    ENDVERBATIM\n"
  "    :}\n"
  "    if (t < delay) {\n"
  "        ivar = 0.\n"
  "    }\n"
  "    else { \n"
  "        if (flag1 == 0) {\n"
  "            flag1 = 1\n"
  "            ivar = mean\n"
  "        }\n"
  "        if (t < delay+dur) {\n"
  "            ivar = mean + exp_decay * (ivar-mean) + amp_gauss * normrand(0,1)\n"
  "            : ivar = ivar + (mean - ivar) * dt / tau + stdev * sqrt(2*dt/tau) * normrand(0,1)\n"
  "            : ivar = stdev*ival\n"
  "        }\n"
  "        else {  \n"
  "            ivar = 0.\n"
  "        }\n"
  "    }\n"
  "    \n"
  "}\n"
  ;
#endif
