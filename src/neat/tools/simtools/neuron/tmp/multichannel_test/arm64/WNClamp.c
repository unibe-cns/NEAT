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
 
#define nrn_init _nrn_init__WNClamp
#define _nrn_initial _nrn_initial__WNClamp
#define nrn_cur _nrn_cur__WNClamp
#define _nrn_current _nrn_current__WNClamp
#define nrn_jacob _nrn_jacob__WNClamp
#define nrn_state _nrn_state__WNClamp
#define _net_receive _net_receive__WNClamp 
#define noiseFromRandom noiseFromRandom__WNClamp 
#define seed seed__WNClamp 
 
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
#define delay _p[0]
#define delay_columnindex 0
#define dur _p[1]
#define dur_columnindex 1
#define mean _p[2]
#define mean_columnindex 2
#define stdev _p[3]
#define stdev_columnindex 3
#define i _p[4]
#define i_columnindex 4
#define on _p[5]
#define on_columnindex 5
#define per _p[6]
#define per_columnindex 6
#define ival _p[7]
#define ival_columnindex 7
#define v _p[8]
#define v_columnindex 8
#define _g _p[9]
#define _g_columnindex 9
#define _tsav _p[10]
#define _tsav_columnindex 10
#define _nd_area  *_ppvar[0]._pval
#define donotuse	*_ppvar[2]._pval
#define _p_donotuse	_ppvar[2]._pval
 
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
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_grand(void*);
 static double _hoc_noiseFromRandom(void*);
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
 "grand", _hoc_grand,
 "noiseFromRandom", _hoc_noiseFromRandom,
 "seed", _hoc_seed,
 0, 0
};
#define grand grand_WNClamp
 extern double grand( _threadargsproto_ );
 /* declare global and static user variables */
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "dur", 0, 1e+09,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "delay", "ms",
 "dur", "ms",
 "mean", "nA",
 "stdev", "nA",
 "i", "nA",
 0,0
};
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void _ba1(Node*_nd, double* _pp, Datum* _ppd, Datum* _thread, NrnThread* _nt) ;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 static void _hoc_destroy_pnt(void* _vptr) {
   destroy_point_process(_vptr);
}
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"WNClamp",
 "delay",
 "dur",
 "mean",
 "stdev",
 0,
 "i",
 0,
 0,
 "donotuse",
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
 	_p = nrn_prop_data_alloc(_mechtype, 11, _prop);
 	/*initialize range parameters*/
 	delay = 0;
 	dur = 0;
 	mean = 0;
 	stdev = 1;
  }
 	_prop->param = _p;
 	_prop->param_size = 11;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
 
#define _tqitem &(_ppvar[3]._pvoid)
 static void _net_receive(Point_process*, double*, double);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _WNClamp_reg() {
	int _vectorized = 1;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 1,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 11, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "pointer");
  hoc_register_dparam_semantics(_mechtype, 3, "netsend");
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 1;
 	hoc_reg_ba(_mechtype, _ba1, 11);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 WNClamp /Users/wybo/Code/NEAT_public/neat/tools/simtools/neuron/tmp/multichannel_test/mech/WNClamp.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int noiseFromRandom(_threadargsproto_);
static int seed(_threadargsprotocomma_ double);
 
static int  seed ( _threadargsprotocomma_ double _lx ) {
   set_seed ( _lx ) ;
    return 0; }
 
static double _hoc_seed(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r = 1.;
 seed ( _p, _ppvar, _thread, _nt, *getarg(1) );
 return(_r);
}
 /* BEFORE BREAKPOINT */
 static void _ba1(Node*_nd, double* _pp, Datum* _ppd, Datum* _thread, NrnThread* _nt)  {
   double* _p; Datum* _ppvar; _p = _pp; _ppvar = _ppd;
  v = NODEV(_nd);
 i = - ival ;
   }
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{  double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _thread = (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;   _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t;   if (_lflag == 1. ) {*(_tqitem) = 0;}
 {
   if ( dur > 0.0 ) {
     if ( _lflag  == 1.0 ) {
       if ( on  == 0.0 ) {
         on = 1.0 ;
         net_send ( _tqitem, _args, _pnt, t +  dur , 1.0 ) ;
         ival = stdev * grand ( _threadargs_ ) + mean ;
         net_send ( _tqitem, _args, _pnt, t +  per , 2.0 ) ;
         }
       else {
         if ( on  == 1.0 ) {
           on = 0.0 ;
           ival = 0.0 ;
           }
         }
       }
     if ( _lflag  == 2.0 ) {
       if ( on  == 1.0 ) {
         ival = stdev * grand ( _threadargs_ ) + mean ;
         net_send ( _tqitem, _args, _pnt, t +  per , 2.0 ) ;
         }
       }
     }
   } }
 
/*VERBATIM*/
double nrn_random_pick(void* r);
void* nrn_random_arg(int argpos);
 
double grand ( _threadargsproto_ ) {
   double _lgrand;
 
/*VERBATIM*/
    if (_p_donotuse) {
        /*
         : Supports separate independent but reproducible streams for
         : each instance. However, the corresponding hoc Random
         : distribution MUST be set to Random.uniform(0,1)
         */
//            _lerand = nrn_random_pick(_p_donotuse);
//            _lurand = nrn_random_pick(_p_donotuse);
            _lgrand = nrn_random_pick(_p_donotuse);
    }else{
        /* only can be used in main thread */
        if (_nt != nrn_threads) {
hoc_execerror("multithread random in InUnif"," only via hoc Random");
        }
 _lgrand = normrand ( 0.0 , 1.0 ) ;
   
/*VERBATIM*/
    }
 
return _lgrand;
 }
 
static double _hoc_grand(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  grand ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
static int  noiseFromRandom ( _threadargsproto_ ) {
   
/*VERBATIM*/
 {
    void** pv = (void**)(&_p_donotuse);
    if (ifarg(1)) {
        *pv = nrn_random_arg(1);
    }else{
        *pv = (void*)0;
    }
 }
  return 0; }
 
static double _hoc_noiseFromRandom(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r = 1.;
 noiseFromRandom ( _p, _ppvar, _thread, _nt );
 return(_r);
}

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
 {
   per = dt ;
   on = 0.0 ;
   ival = 0.0 ;
   i = 0.0 ;
   net_send ( _tqitem, (double*)0, _ppvar[1]._pvoid, t +  delay , 1.0 ) ;
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

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/wybo/Code/NEAT_public/neat/tools/simtools/neuron/tmp/multichannel_test/mech/WNClamp.mod";
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
  "    POINT_PROCESS WNClamp\n"
  "    NONSPECIFIC_CURRENT i\n"
  "    RANGE mean, stdev\n"
  "    RANGE delay, dur\n"
  "    THREADSAFE : true only if every instance has its own distinct Random\n"
  "    POINTER donotuse\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "    (nA) = (nanoamp)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "    delay (ms) : delay until noise starts\n"
  "    dur (ms) <0, 1e9> : duration of noise\n"
  "    mean = 0 (nA)\n"
  "    stdev = 1 (nA)\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "    dt (ms)\n"
  "    on\n"
  "    per (ms)\n"
  "    ival (nA)\n"
  "    i (nA)\n"
  "    donotuse\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "    per = dt\n"
  "    on = 0\n"
  "    ival = 0\n"
  "    i = 0\n"
  "    net_send(delay, 1)\n"
  "}\n"
  "\n"
  "PROCEDURE seed(x) {\n"
  "    set_seed(x)\n"
  "}\n"
  "\n"
  "BEFORE BREAKPOINT {\n"
  "    i = -ival\n"
  ": printf(\"time %f \\ti %f\\n\", t, ival)\n"
  "}\n"
  "\n"
  "BREAKPOINT { : this block must exist so that a current is actually generated\n"
  "}\n"
  "\n"
  "NET_RECEIVE (w) {\n"
  "    if (dur>0) {\n"
  "        if (flag==1) {\n"
  "            if (on==0) { : turn on\n"
  "                on=1\n"
  "                net_send(dur,1) : to turn it off\n"
  ":                ival = (hi-lo)*urand() + lo : first sample\n"
  "                ival = stdev*grand() + mean : first sample\n"
  "                net_send(per, 2) : prepare for next sample\n"
  "            } else {\n"
  "                if (on==1) { : turn off\n"
  "                    on=0\n"
  "                    ival = 0\n"
  "                }\n"
  "            }\n"
  "        }\n"
  "        if (flag==2) {\n"
  "            if (on==1) {\n"
  "                ival = stdev*grand() + mean\n"
  ": printf(\"time %f \\ti %f\\n\", t, ival)\n"
  "                net_send(per, 2) : prepare for next sample\n"
  "            }\n"
  "        }\n"
  "    }\n"
  "}\n"
  "\n"
  "VERBATIM\n"
  "double nrn_random_pick(void* r);\n"
  "void* nrn_random_arg(int argpos);\n"
  "ENDVERBATIM\n"
  "\n"
  ": FUNCTION erand() {\n"
  ": FUNCTION urand() {\n"
  "FUNCTION grand() {\n"
  "VERBATIM\n"
  "    if (_p_donotuse) {\n"
  "        /*\n"
  "         : Supports separate independent but reproducible streams for\n"
  "         : each instance. However, the corresponding hoc Random\n"
  "         : distribution MUST be set to Random.uniform(0,1)\n"
  "         */\n"
  "//            _lerand = nrn_random_pick(_p_donotuse);\n"
  "//            _lurand = nrn_random_pick(_p_donotuse);\n"
  "            _lgrand = nrn_random_pick(_p_donotuse);\n"
  "    }else{\n"
  "        /* only can be used in main thread */\n"
  "        if (_nt != nrn_threads) {\n"
  "hoc_execerror(\"multithread random in InUnif\",\" only via hoc Random\");\n"
  "        }\n"
  "ENDVERBATIM\n"
  "        : the old standby. Cannot use if reproducible parallel sim\n"
  "        : independent of nhost or which host this instance is on\n"
  "        : is desired, since each instance on this cpu draws from\n"
  "        : the same stream\n"
  ":        erand = exprand(1)\n"
  ":        urand = scop_random()\n"
  "        grand = normrand(0,1)\n"
  ": printf(\"%f\\n\", grand)\n"
  "VERBATIM\n"
  "    }\n"
  "ENDVERBATIM\n"
  "}\n"
  "\n"
  "PROCEDURE noiseFromRandom() {\n"
  "VERBATIM\n"
  " {\n"
  "    void** pv = (void**)(&_p_donotuse);\n"
  "    if (ifarg(1)) {\n"
  "        *pv = nrn_random_arg(1);\n"
  "    }else{\n"
  "        *pv = (void*)0;\n"
  "    }\n"
  " }\n"
  "ENDVERBATIM\n"
  "}\n"
  ;
#endif
